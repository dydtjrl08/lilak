"""Truth-free 3-D line-finding baselines for observed TPC pad/time samples.

The input is (x_pad, y_layer, z_time) in mm, not MC positions. Fractions use
alpha slots first and background LAST, like the accompanying CNN. Fixed K is
a *conditional multiplicity* benchmark; these routines do not estimate K.

Hough really votes in a direction/intercept accumulator: for each candidate
3-D direction, points are projected onto its orthogonal plane and binned in
two intercept coordinates. RANSAC instead samples pairs and scores lines.
Both subsequently use the same weighted TLS and soft-assignment refinement.

``scales_mm`` specifies a DIAGNOSTIC RESIDUAL METRIC, not a measured detector
resolution or a calibrated diffusion law. Its defaults reflect anisotropic
pad sampling and a deliberately approximate time width. Tune these on train /
validation only. Max/threshold/coarse-time sampling can lose physical charge
information; geometric fractions are proximity weights, NOT measured charge
purities. In particular, identical coincident tracks cannot be separated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class Config:
    scales_mm: tuple = (1.5, 6.0, 2.5)
    proposal_points: int = 1000
    hough_directions: int = 192
    hough_pair_directions: int = 64
    hough_binsize: float = 1.5
    hough_offsets: int = 2
    intercept_peaks: int = 2
    ransac_trials: int = 256
    max_candidates: int = 256
    inlier_radius: float = 2.0
    assignment_width: float = 1.0
    background_score: float = 0.08
    min_points: int = 4
    min_layers: int = 2
    min_span: float = 2.0
    min_remaining_fraction: float = 0.025
    duplicate_angle_deg: float = 6.0
    duplicate_distance: float = 1.0
    refinement_iterations: int = 5
    prior_power: float = 1.0


def _config(config):
    if config is None:
        return Config()
    if isinstance(config, Config):
        return config
    config = dict(config)
    # Useful spelling when a caller tunes an axis metric in a small grid.
    if "scales" in config:
        config["scales_mm"] = config.pop("scales")
    return Config(**config)


def _input(points, charge, config):
    points = np.asarray(points, dtype=np.float64)
    charge = np.asarray(charge, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3), in pad xyz mm")
    if charge.shape != (len(points),):
        raise ValueError("charge must have shape (N,)")
    scales = np.asarray(config.scales_mm, dtype=np.float64)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales_mm must contain three finite positive mm scales")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(charge)):
        raise ValueError("points and charge must be finite")
    if config.assignment_width <= 0 or config.inlier_radius <= 0 or config.hough_binsize <= 0:
        raise ValueError("geometric widths must be positive")
    if not 0 < config.background_score < 1:
        raise ValueError("background_score must be between 0 and 1")
    # sqrt prevents the brightest pulse alone defining all proposals; it is
    # an explicit robustness choice, shared by both geometric methods.
    weight = np.sqrt(np.maximum(charge, 0))
    positive = weight > 0
    if np.any(positive):
        weight /= np.median(weight[positive])
    return points, points / scales, weight, scales


def _canonical(direction):
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None
    direction = direction / norm
    # An undirected line: canonical sign makes permutation/sign tests stable.
    if direction[int(np.argmax(np.abs(direction)))] < 0:
        direction = -direction
    return direction


def _tls(points, weights):
    valid = weights > 1e-12
    if np.count_nonzero(valid) < 2:
        return None
    p, w = points[valid], weights[valid]
    center = np.average(p, axis=0, weights=w)
    delta = p - center
    covariance = (delta * w[:, None]).T @ delta / w.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1e-12:
        return None
    return center, _canonical(eigenvectors[:, -1])


def _distance(points, origin, direction):
    delta = points - origin
    longitudinal = delta @ direction
    return np.sqrt(np.maximum(np.einsum("ij,ij->i", delta, delta) - longitudinal**2, 0))


def _support(points, line, weights, config):
    distance = _distance(points, *line)
    mask = (distance <= config.inlier_radius) & (weights > 0)
    if np.count_nonzero(mask) < config.min_points:
        return False
    # y is a true pad-layer coordinate, scaled but still exactly repeated.
    if len(np.unique(points[mask, 1])) < config.min_layers:
        return False
    longitudinal = (points[mask] - line[0]) @ line[1]
    if np.ptp(longitudinal) < config.min_span:
        return False
    return True


def _proposal_sample(points, weights, config, rng):
    indices = np.flatnonzero(weights > 0)
    if len(indices) > config.proposal_points:
        probability = weights[indices] / weights[indices].sum()
        # An explicit uniform component preserves weak/tail geometry.
        probability = 0.75 * probability + 0.25 / len(indices)
        indices = rng.choice(indices, config.proposal_points, replace=False, p=probability)
        # Sort input lexicographically before sampling in infer; indices then
        # have reproducible meanings even if caller permutes input rows.
        indices.sort()
    return points[indices], weights[indices]


def _sample_directions(points, count, rng):
    if len(points) < 2:
        return np.empty((0, 3))
    first = rng.integers(0, len(points), size=count * 4)
    second = rng.integers(0, len(points), size=count * 4)
    delta = points[second] - points[first]
    norm = np.linalg.norm(delta, axis=1)
    good = (norm > 1e-8) & (np.abs(delta[:, 1]) > 1e-8)
    delta = delta[good][:count]
    norm = np.linalg.norm(delta, axis=1)
    return delta / norm[:, None]


def _sphere_directions(count):
    # A hemisphere is sufficient because a line is undirected.
    i = np.arange(count, dtype=float) + 0.5
    z = i / max(count, 1)
    azimuth = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def _hough(points, weights, config, rng):
    directions = np.vstack((
        _sphere_directions(config.hough_directions),
        _sample_directions(points, config.hough_pair_directions, rng),
        np.eye(3),
    ))
    candidates = []
    for direction in directions:
        # Two orthonormal coordinates parameterize a line's intercept plane.
        seed_axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        first = np.cross(direction, seed_axis)
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        projection = np.column_stack((points @ first, points @ second))
        for offset in range(config.hough_offsets):
            # Shift accumulator boundaries; this is still genuine voting.
            shift = offset / max(config.hough_offsets, 1)
            bins = np.floor(projection / config.hough_binsize + shift).astype(np.int64)
            unique, inverse = np.unique(bins, axis=0, return_inverse=True)
            votes = np.bincount(inverse, weights=weights, minlength=len(unique))
            number = min(config.intercept_peaks, len(votes))
            if not number:
                continue
            peaks = np.argsort(votes, kind="stable")[-number:][::-1]
            for peak in peaks:
                members = inverse == peak
                if np.count_nonzero(members) < config.min_points:
                    continue
                intercept = np.average(projection[members], axis=0, weights=weights[members])
                origin = intercept[0] * first + intercept[1] * second
                candidates.append((votes[peak], origin, _canonical(direction)))
    # Accumulator vote preselection keeps CPU use bounded; final support is
    # rescored consistently with RANSAC on the same proposal sample.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(origin, direction) for _, origin, direction in candidates[:config.max_candidates]]


def _ransac(points, weights, config, rng):
    candidates = []
    if len(points) < 2:
        return candidates
    for _ in range(config.ransac_trials):
        indices = rng.choice(len(points), 2, replace=False)
        first, second = points[indices]
        if abs(first[1] - second[1]) <= 1e-8:
            continue  # same-pad waveform streak is not a longitudinal track
        direction = _canonical(second - first)
        if direction is not None:
            candidates.append((first, direction))
    return candidates


def _duplicate(first, second, config):
    cosine = np.clip(abs(first[1] @ second[1]), 0, 1)
    if np.degrees(np.arccos(cosine)) > config.duplicate_angle_deg:
        return False
    # For approximately parallel lines, origin-to-line distance is useful;
    # requiring both distances avoids merging distinct crossing directions.
    return (float(_distance(first[0][None], *second)[0]) < config.duplicate_distance
            and float(_distance(second[0][None], *first)[0]) < config.duplicate_distance)


def _select(points, weights, candidates, n_tracks, config):
    selected, residual_weight = [], weights.copy()
    candidates = list(candidates)
    for _ in range(n_tracks):
        best, best_score = None, 0.0
        for proposal in candidates:
            if not _support(points, proposal, residual_weight, config):
                continue
            distance = _distance(points, *proposal)
            fit_weight = residual_weight * np.exp(-0.5 * (distance / config.assignment_width)**2)
            fit_weight[distance > config.inlier_radius] = 0
            fitted = _tls(points, fit_weight)
            if fitted is None or not _support(points, fitted, residual_weight, config):
                continue
            if any(_duplicate(fitted, old, config) for old in selected):
                continue
            distance = _distance(points, *fitted)
            kernel = np.exp(-0.5 * (distance / config.assignment_width)**2)
            kernel[distance > config.inlier_radius] = 0
            score = float(residual_weight @ kernel)
            if score > best_score:
                best, best_score = fitted, score
        if best is None or best_score < config.min_remaining_fraction * residual_weight.sum():
            break
        selected.append(best)
        distance = _distance(points, *best)
        # Peel softly so ambiguous overlap samples remain available.
        residual_weight *= 1 - np.exp(-0.5 * (distance / config.assignment_width)**2)
    return selected


def _assignment(points, weights, lines, n_tracks, config, prior=None):
    scores = np.zeros((len(points), n_tracks + 1), dtype=float)
    for slot, line in enumerate(lines):
        if line is None:
            continue
        distance = _distance(points, *line)
        scores[:, slot] = np.exp(-0.5 * (distance / config.assignment_width)**2)
    scores[:, -1] = config.background_score
    if prior is not None:
        # A small floor allows geometry to correct an overconfident CNN;
        # this is a configurable prior-product, not truth charge recovery.
        scores *= np.maximum(prior, 1e-4)**config.prior_power
    scores /= scores.sum(axis=1, keepdims=True)
    scores[weights <= 0] = 0
    scores[weights <= 0, -1] = 1
    return scores


def _iterate(points, weights, lines, n_tracks, config, prior=None):
    lines = list(lines) + [None] * (n_tracks - len(lines))
    for _ in range(config.refinement_iterations):
        fractions = _assignment(points, weights, lines, n_tracks, config, prior)
        for slot, line in enumerate(lines):
            if line is None:
                continue
            distance = _distance(points, *line)
            fit_weight = weights * fractions[:, slot]
            fit_weight[distance > config.inlier_radius] = 0
            fitted = _tls(points, fit_weight)
            if fitted is not None and _support(points, fitted, fit_weight, config):
                lines[slot] = fitted
    return lines, _assignment(points, weights, lines, n_tracks, config, prior)


def _physical_lines(lines, scales, n_tracks):
    result = np.full((n_tracks, 6), np.nan, dtype=np.float64)
    for slot, line in enumerate(lines):
        if line is not None:
            result[slot, :3] = line[0] * scales
            result[slot, 3:] = _canonical(line[1] * scales)
    return result


def infer(points, charge, n_tracks, method="hough", seed=0, config=None):
    """Infer line/proximity fractions using OBSERVATIONS ONLY.

    Returns fractions (N,K+1; background last), lines (K,6; origin mm +
    unit physical direction), found_count, seconds, and config. Missing slots
    have NaN lines and zero fractions; no fabricated Kth line is forced.
    """
    start, cfg = perf_counter(), _config(config)
    if not isinstance(n_tracks, (int, np.integer)) or n_tracks < 1:
        raise ValueError("n_tracks must be a positive integer")
    if method not in {"hough", "ransac"}:
        raise ValueError("method must be 'hough' or 'ransac'")
    physical, metric, weights, scales = _input(points, charge, cfg)
    # Canonical sorting makes output independent of arbitrary sample order.
    order = np.lexsort((weights, metric[:, 2], metric[:, 1], metric[:, 0]))
    inverse = np.argsort(order)
    metric, weights = metric[order], weights[order]
    rng = np.random.default_rng(seed)
    sample, sample_weight = _proposal_sample(metric, weights, cfg, rng)
    candidates = (_hough(sample, sample_weight, cfg, rng) if method == "hough"
                  else _ransac(sample, sample_weight, cfg, rng))
    selected = _select(sample, sample_weight, candidates, n_tracks, cfg)
    lines, fractions = _iterate(metric, weights, selected, n_tracks, cfg)
    return {
        "fractions": fractions[inverse].astype(np.float32),
        "lines": _physical_lines(lines, scales, n_tracks),
        "found_count": sum(line is not None for line in lines),
        "seconds": perf_counter() - start,
        "method": method,
        "config": asdict(cfg),
    }


def refine(points, charge, prior_fractions, lines=None, config=None):
    """CNN/geometric hybrid; no MC labels accepted by this interface.

    With lines=None, start with each CNN slot's weighted TLS. Alternatively
    pass infer(...)["lines"] for Hough -> CNN-prior refinement. Geometric
    slots are matched to CNN slots by maximum soft observation overlap, NOT
    truth. For the present 2/3-alpha problem exhaustive permutations are tiny.
    """
    start, cfg = perf_counter(), _config(config)
    physical, metric, weights, scales = _input(points, charge, cfg)
    prior = np.asarray(prior_fractions, dtype=float)
    if prior.ndim != 2 or prior.shape[0] != len(points) or prior.shape[1] < 2:
        raise ValueError("prior_fractions must have shape (N,K+1), background last")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0):
        raise ValueError("prior_fractions must be finite and nonnegative")
    prior = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)
    prior[prior.sum(axis=1) == 0, -1] = 1
    n_tracks = prior.shape[1] - 1
    if lines is None:
        metric_lines = []
        for slot in range(n_tracks):
            fit_weight = weights * prior[:, slot]**2
            line = _tls(metric, fit_weight)
            metric_lines.append(line if line is not None and _support(metric, line, fit_weight, cfg) else None)
    else:
        lines = np.asarray(lines, dtype=float)
        if lines.shape != (n_tracks, 6):
            raise ValueError("lines must have shape (K,6)")
        metric_lines = []
        for line in lines:
            metric_lines.append((line[:3] / scales, _canonical(line[3:] / scales))
                                if np.all(np.isfinite(line)) else None)
        if n_tracks > 6:
            raise ValueError("hybrid geometric-slot matching supports at most 6 tracks")
        geometric = _assignment(metric, weights, metric_lines, n_tracks, cfg)[:, :n_tracks]
        overlap = geometric.T @ (weights[:, None] * prior[:, :n_tracks])
        # permutation[j] is the geometric slot assigned to CNN slot j.
        best = max(permutations(range(n_tracks)),
                   key=lambda p: sum(overlap[p[j], j] for j in range(n_tracks)))
        metric_lines = [metric_lines[j] for j in best]
    metric_lines, fractions = _iterate(metric, weights, metric_lines, n_tracks, cfg, prior)
    return {
        "fractions": fractions.astype(np.float32),
        "lines": _physical_lines(metric_lines, scales, n_tracks),
        "found_count": sum(line is not None for line in metric_lines),
        "seconds": perf_counter() - start,
        "method": "cnn_prior_geometric_refinement",
        "config": asdict(cfg),
    }
