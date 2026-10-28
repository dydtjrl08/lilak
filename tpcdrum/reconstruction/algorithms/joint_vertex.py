"""Common-origin regions and joint fits of ORIGINAL observation measurements.

This is profile/generalized least squares, optionally with explicit angular-kick
penalties. It is NOT a full Kalman fitter. No pair-derived vertex is introduced
as a second measurement. Numerical scan limits are not a reaction volume, and
neither a first-hit centre nor a reaction-volume boundary clamps the estimate.
All uncertainty is conditional on the supplied covariance/process model; this
module establishes no actual detector-resolution or MC-production-angle claim.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq, least_squares, minimize_scalar


def _require(ok, text):
    if not ok:
        raise ValueError(text)


def _axes(dimension):
    _require(dimension in (2, 3), "dimension must be 2 (u,v) or 3 (u,v,w)")
    return [0] if dimension == 2 else [0, 2]


def _covariance(matrix, size, name):
    matrix = np.asarray(matrix, dtype=float)
    _require(matrix.shape == (size, size) and np.isfinite(matrix).all(), name + " shape/finite check failed")
    scales = np.sqrt(np.maximum(np.abs(np.diag(matrix)), 1e-30))
    normalized = matrix / scales[:, None] / scales[None, :]
    _require(np.allclose(normalized, normalized.T, rtol=1e-8, atol=1e-9) and
             np.linalg.eigvalsh((normalized + normalized.T) / 2.)[0] >= -1e-9,
             name + " is not symmetric positive semidefinite")
    return (matrix + matrix.T) / 2.


def _white(residual, covariance):
    covariance = (covariance + covariance.T) / 2.
    try:
        return np.linalg.solve(np.linalg.cholesky(covariance), residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("Projected covariance is singular; no invented precision floor was added") from error


def _provenance(config, k, dimension):
    if dimension == 3:
        _require(config.get("physical_3d") is True,
                 "Physical 3D requires explicit calibrated physical_3d=true; timing-feature coordinates are not w")
    groups = config.get("track_source_groups")
    fragments = config.get("track_fragment_ids")
    if groups is not None:
        _require(len(groups) == k and all(x == groups[0] for x in groups),
                 "Cross-origin/source-event track combination rejected")
    if fragments is not None:
        _require(len(fragments) == k and len(set(map(str, fragments))) == k,
                 "Duplicate observation fragment rejected")
    return dict(same_source_group_verified=groups is not None,
                distinct_fragment_ids_verified=fragments is not None,
                missing_provenance_is_not_a_common_origin_verification=groups is None or fragments is None)


def _soft_sigmas(config, k):
    value = config.get("first_layer_v_sigma_mm")
    source = "explicit first-layer depth uncertainty"
    if value is None and config.get("first_layer_half_extent_mm") is not None:
        value = np.asarray(config["first_layer_half_extent_mm"], dtype=float) / math.sqrt(3.)
        source = "explicit uniform layer-extent sampling proxy; not calibrated detector uncertainty"
    if value is None:
        return None, "no first-layer bound; no hard clamp"
    sigma = np.broadcast_to(np.asarray(value, dtype=float), (k,)).copy()
    _require(np.isfinite(sigma).all() and np.all(sigma > 0.), "Positive first-layer sigma/extent required")
    return sigma, source


def _shared_offset(config, dimension):
    value = config.get("shared_w_offset")
    if value is None:
        return 0.
    _require(dimension == 3, "A shared w/clock offset cannot be applied to projected 2D")
    _require(value.get("included_in_measurement_covariance") is False and value.get("source"),
             "Shared offset needs an explicit independent source and covariance exclusion declaration")
    sigma = float(value.get("sigma_mm", np.nan))
    _require(np.isfinite(sigma) and sigma > 0., "shared_w_offset.sigma_mm must be positive")
    return sigma


def _external(config, dimension):
    value = config.get("external_vertex_prior")
    beam = config.get("external_beam_line")
    for item in (value, beam):
        if item is not None:
            _require(item.get("independent") is True and item.get("derived_from_current_tracks") is False
                     and bool(item.get("source")),
                     "External constraint must be an explicit independent observation; a track-derived vertex is not a new prior")
    if value is not None:
        value = copy.deepcopy(value)
        value["mean"] = np.asarray(value.get("mean"), dtype=float)
        _require(value["mean"].shape == (dimension,) and np.isfinite(value["mean"]).all(), "Invalid external vertex mean")
        value["covariance"] = _covariance(value.get("covariance"), dimension, "external vertex covariance")
        _white(np.zeros(dimension), value["covariance"])
    if beam is not None:
        beam = copy.deepcopy(beam)
        beam["origin"] = np.asarray(beam.get("origin"), dtype=float)
        beam["slopes"] = np.asarray(beam.get("slopes"), dtype=float)
        _require(beam["origin"].shape == (dimension,) and beam["slopes"].shape == (dimension - 1,) and
                 np.isfinite(beam["origin"]).all() and np.isfinite(beam["slopes"]).all(), "Invalid independent beam line")
        beam["covariance"] = _covariance(beam.get("state_covariance"), 2 * (dimension - 1), "beam state covariance")
    return value, beam


def _extra_residual(vertex, references, sigma, external, beam, axes):
    result = []
    if sigma is not None:
        result.extend(np.maximum(vertex[1] - references, 0.) / sigma)
    if external is not None:
        result.extend(_white(vertex - external["mean"], external["covariance"]))
    if beam is not None:
        dv = vertex[1] - beam["origin"][1]
        m = len(axes); projection = np.c_[np.eye(m), np.eye(m) * dv]
        covariance = projection @ beam["covariance"] @ projection.T
        result.extend(_white(vertex[axes] - beam["origin"][axes] - beam["slopes"] * dv, covariance))
    return np.asarray(result, dtype=float)


def _reaction(config, dimension):
    volume = config.get("reaction_volume")
    if volume is None:
        return None
    _require(bool(volume.get("source")), "Explicit reaction volume needs its actual geometry source")
    bounds = np.asarray(volume.get("bounds_uvw_mm", volume.get("bounds_uv_mm")), dtype=float)
    if dimension == 2 and bounds.shape == (2, 3):
        bounds = bounds[:, :2]
    _require(bounds.shape == (2, dimension) and np.isfinite(bounds).all() and np.all(bounds[1] > bounds[0]),
             "Invalid explicit reaction-volume bounds")
    return bounds


def _scan(config, volume):
    scan = config.get("vertex_v_scan_mm")
    if scan is None and volume is not None:
        scan = volume[:, 1]
    _require(scan is not None, "Missing explicit numerical vertex_v_scan_mm or sourced reaction volume")
    scan = np.asarray(scan, dtype=float)
    _require(scan.shape == (2,) and np.isfinite(scan).all() and scan[1] > scan[0], "Invalid numerical vertex scan")
    return scan


def _finite_jacobian(function, point):
    y = np.asarray(function(point), dtype=float)
    result = np.empty((len(y), len(point)))
    for j in range(len(point)):
        h = max(1e-5, abs(point[j]) * 1e-6)
        a, b = point.copy(), point.copy(); a[j] += h; b[j] -= h
        result[:, j] = (function(a) - function(b)) / (2. * h)
    return result


def _information_covariance(jacobian, condition_limit):
    norm = np.linalg.norm(jacobian, axis=0)
    safe = np.where(norm > 0., norm, 1.)
    normalized = jacobian / safe
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    tolerance = singular[0] / float(condition_limit) if len(singular) else 1.
    rank = int(np.sum(singular > max(tolerance, 1e-12)))
    full = rank == jacobian.shape[1]
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0. else float("inf")
    covariance = np.full((jacobian.shape[1], jacobian.shape[1]), np.nan)
    if full:
        covariance = ((vt.T / singular ** 2) @ vt) / safe[:, None] / safe[None, :]
    return covariance, rank, condition


def _direction(slopes, dimension):
    slopes = np.asarray(slopes)
    result = np.ones(slopes.shape[:-1] + (dimension,))
    result[..., 0] = slopes[..., 0]
    if dimension == 3:
        result[..., 2] = slopes[..., 1]
    return result / np.linalg.norm(result, axis=-1)[..., None]


def _angular_covariance(slopes, variance):
    slopes = np.asarray(slopes)
    return variance * (1. + slopes @ slopes) * (np.eye(len(slopes)) + np.outer(slopes, slopes))


def vertex_region(origins, directions, state_covariances, reference_v, config, dimension=3):
    """Profile a common region from local states and their FULL covariance.

    3D arrays: origins/directions Kx3=(u,v,w), cov Kx4x4=(u,w,tu,tw), or
    Kx5x5 (energy marginalized away). 2D: Kx2 and cov Kx2x2=(u,tu).
    A full (K*2m)x(K*2m) state covariance also preserves cross-track correlations.
    None is returned as the point when v is scan-censored/near-parallel. The
    envelope remains useful for a third-track proposal, without a fake vertex.
    """
    result = dict(status="invalid_input", success=False, point=None, covariance=None,
                  v_interval_mm=None, open_v_sides=[True, True], dimension=dimension,
                  physical_uncertainty_validated=False, physics_accuracy_evaluated=False)
    try:
        axes = _axes(dimension); m = len(axes)
        origins, directions = np.asarray(origins, float), np.asarray(directions, float)
        references = np.asarray(reference_v, float)
        k = len(origins)
        _require(k >= 2 and origins.shape == (k, dimension) and directions.shape == origins.shape and
                 references.shape == (k,) and np.isfinite(origins).all() and np.isfinite(directions).all(),
                 "At least two finite local track states are required")
        provenance = _provenance(config, k, dimension)
        _require(np.allclose(origins[:, 1], references, atol=1e-8, rtol=0.), "Local-state reference_v disagrees with origin plane")
        _require(np.all(directions[:, 1] > 1e-10) and np.all(np.linalg.norm(directions, axis=1) > 0.),
                 "Missing/invalid forward local starting direction")
        slopes = directions[:, axes] / directions[:, 1, None]
        covariance = np.asarray(state_covariances, float)
        if dimension == 3 and covariance.shape == (k, 5, 5):
            covariance = covariance[:, :4, :4]
        if covariance.shape == (k, 2 * m, 2 * m):
            covariance = block_diag(*[_covariance(c, 2 * m, "local state covariance") for c in covariance])
        else:
            covariance = _covariance(covariance, k * 2 * m, "joint local-state covariance")
        sigma, sigma_source = _soft_sigmas(config, k)
        shared = _shared_offset(config, dimension)
        external, beam = _external(config, dimension)
        volume = _reaction(config, dimension); scan = _scan(config, volume)
        q = float(config.get("unobserved_angular_variance_rad2_per_mm", 0.))
        _require(np.isfinite(q) and q >= 0., "Invalid explicit unobserved angular process noise")

        def at(v):
            projection = np.zeros((k * m, k * 2 * m))
            values = []
            for i in range(k):
                dv = v - references[i]
                projection[i*m:(i+1)*m, i*2*m:(i+1)*2*m] = np.c_[np.eye(m), np.eye(m)*dv]
                values.extend(origins[i, axes] + slopes[i] * dv)
            cov = projection @ covariance @ projection.T
            if q:
                for i in range(k):
                    dv = v - references[i]
                    length = abs(dv) * math.sqrt(1. + slopes[i] @ slopes[i])
                    cov[i*m:(i+1)*m, i*m:(i+1)*m] += dv ** 2 / 3. * _angular_covariance(slopes[i], q * length)
            if shared:
                shift = np.zeros(k * m); shift[m-1::m] = 1.
                cov += shared ** 2 * np.outer(shift, shift)
            h = np.tile(np.eye(m), (k, 1))
            inv_h = np.linalg.solve(cov, h)
            information = h.T @ inv_h
            rhs = inv_h.T @ np.asarray(values)
            penalty = 0.
            if external is not None:
                c = external["covariance"]; cv = c[1, 1]
                _require(cv > 0., "External vertex prior requires nonzero depth variance")
                conditional_mean = external["mean"][axes] + c[np.ix_(axes, [1])].ravel()/cv * (v-external["mean"][1])
                conditional_cov = c[np.ix_(axes, axes)] - np.outer(c[axes, 1], c[1, axes])/cv
                inv = np.linalg.inv(conditional_cov); information += inv; rhs += inv @ conditional_mean
                penalty += (v-external["mean"][1]) ** 2 / cv
            if beam is not None:
                dv = v - beam["origin"][1]; a = np.c_[np.eye(m), np.eye(m)*dv]
                bc = a @ beam["covariance"] @ a.T
                inv = np.linalg.inv(bc); information += inv
                rhs += inv @ (beam["origin"][axes] + beam["slopes"]*dv)
            transverse_cov = np.linalg.inv(information)
            center = transverse_cov @ rhs
            vertex = np.empty(dimension); vertex[1] = v; vertex[axes] = center
            residual = h @ center - values
            score = float(residual @ np.linalg.solve(cov, residual))
            extra = _extra_residual(vertex, references, sigma, external, beam, axes)
            # external residuals, including the full correlated covariance, are
            # evaluated directly; conditional penalties above only find center.
            score += float(extra @ extra)
            return score, vertex, transverse_cov, cov

        count = int(config.get("scan_points", 257)); _require(count >= 17, "scan_points must be >=17")
        grid = np.linspace(scan[0], scan[1], count)
        scores = np.array([at(v)[0] for v in grid])
        candidates = [(float(scores[0]), float(grid[0])), (float(scores[-1]), float(grid[-1]))]
        for i in range(1, count-1):
            if (scores[i] <= scores[i-1] and scores[i] <= scores[i+1] and
                    min(scores[i-1]-scores[i], scores[i+1]-scores[i]) > 1e-12):
                optimum = minimize_scalar(lambda v: at(v)[0], bounds=(grid[i-1], grid[i+1]), method="bounded",
                                          options={"xatol": 1e-8})
                candidates.append((float(optimum.fun), float(optimum.x)))
        minimum, best_v = min(candidates)
        best = at(best_v)[1]
        threshold = float(config.get("profile_delta_chi2", 3.841458820694124))
        _require(np.isfinite(threshold) and threshold > 0., "Positive profile_delta_chi2 required")
        grid = np.sort(np.unique(np.r_[grid, best_v]))
        scores = np.array([at(v)[0] for v in grid])
        inside = scores <= minimum + threshold
        intervals = []
        for a in np.flatnonzero(inside & np.r_[True, ~inside[:-1]]):
            b = a
            while b + 1 < len(grid) and inside[b+1]:b += 1
            lo = grid[a] if a == 0 else brentq(lambda v: at(v)[0]-minimum-threshold, grid[a-1], grid[a])
            hi = grid[b] if b == len(grid)-1 else brentq(lambda v: at(v)[0]-minimum-threshold, grid[b], grid[b+1])
            intervals.append([float(lo), float(hi)])
        open_sides = [bool(inside[0]), bool(inside[-1])]
        # Full point residual Jacobian keeps position/angle and cross-track
        # correlations. It is a local Gaussian approximation, not a calibration.
        def residual(vertex):
            _, _, _, c = at(vertex[1])
            r = np.concatenate([vertex[axes]-origins[i, axes]-slopes[i]*(vertex[1]-references[i]) for i in range(k)])
            return np.r_[_white(r, c), _extra_residual(vertex, references, sigma, external, beam, axes)]
        pcov, rank, condition = _information_covariance(_finite_jacobian(residual, best), config.get("condition_limit", 1e10))
        resolved = not any(open_sides) and rank == dimension and np.isfinite(pcov).all()
        env = [at(v) for v in grid]
        result.update(status="resolved" if resolved else "extended_region", success=resolved,
                      point=best if resolved else None, covariance=pcov if resolved else None,
                      candidate_point=best, candidate_covariance=pcov,
                      v_interval_mm=[intervals[0][0], intervals[-1][1]], v_intervals_mm=intervals,
                      open_v_sides=open_sides, profile_minimum_chi2=minimum,
                      transverse_region_envelope=dict(v_samples_mm=grid,
                           centers_mm=np.array([item[1][axes] for item in env]),
                           covariances_mm2=np.array([item[2] for item in env])),
                      quality=dict(rank=rank, condition=condition, **provenance),
                      first_layer_support=dict(applied=sigma is not None, source=sigma_source,
                                               hard_clamp=False, independent_new_measurement=False),
                      scan_semantics="explicit numerical profile scan; not an inferred physical reaction region",
                      local_state_extrapolation_model="straight local tangent + explicit optional angular diffusion",
                      no_120_degree_constraint=True)
        if volume is not None:
            volume_inside = bool(np.all(best >= volume[0]) and np.all(best < volume[1]))
            result["reaction_volume"] = dict(bounds=volume, source=config["reaction_volume"]["source"],
                used_as_hard_constraint=False, used_as_acceptance_guard=True, candidate_inside=volume_inside)
            if not volume_inside:
                result.update(status="outside_reaction_volume", success=False, point=None, covariance=None)
        return result
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        result["message"] = str(error)
        return result


def _measurement_residual(point, fitted_transverse, slope, covariance, axes, offset):
    m = len(axes); dimension = m + 1
    projection = np.zeros((m, dimension))
    projection[np.arange(m), axes] = 1.; projection[:, 1] = -slope
    r = fitted_transverse - point[axes]
    if offset:r[-1] += offset
    return _white(r, projection @ covariance @ projection.T)


def joint_fit(measurement_tracks, covariances, config):
    """Joint common-origin fit and FAIR free fit of the exact same observations.

    process_noise_mode='none': straight generalized least squares. 'tuned' with
    explicit angular_variance_rad2_per_mm>0: piecewise-linear angular-kick
    penalized least squares. The latter is not branded as a Kalman fit.
    Local directions refer to observed nodes. vertex_track_directions are latent
    vertex-to-first-chord directions UNDER THIS MODEL, not recovered MC truth.
    """
    result = dict(status="invalid_input", success=False, vertex=None, vertex_covariance=None,
                  physical_uncertainty_validated=False, physics_accuracy_evaluated=False,
                  no_120_degree_constraint=True)
    try:
        dimension = int(config.get("dimension", 3)); axes = _axes(dimension); m = len(axes)
        tracks = [np.asarray(p, dtype=float) for p in measurement_tracks]
        raw_cov = [np.asarray(c, dtype=float) for c in covariances]
        k = len(tracks); _require(k >= 2 and len(raw_cov) == k, "Two or more original measured tracks required")
        provenance = _provenance(config, k, dimension)
        _require(config.get("vertex_prior_from_current_tracks") is None,
                 "A derived pair vertex cannot be counted as an independent prior")
        for i in range(k):
            _require(tracks[i].ndim == 2 and tracks[i].shape[1] == dimension and len(tracks[i]) >= 3 and
                     np.isfinite(tracks[i]).all(), "Each track needs >=3 finite ORIGINAL measured points")
            _require(raw_cov[i].shape == (len(tracks[i]), dimension, dimension), "Measurement covariance shape mismatch")
            order = np.argsort(tracks[i][:, 1], kind="stable")
            tracks[i] = tracks[i][order]; raw_cov[i] = raw_cov[i][order]
            _require(np.all(np.diff(tracks[i][:, 1]) > 0.), "Duplicate/nonincreasing layer depth in one observation fragment")
            raw_cov[i] = np.array([_covariance(c, dimension, "original measurement covariance") for c in raw_cov[i]])
        references = np.array([p[0, 1] for p in tracks])
        sigma, sigma_source = _soft_sigmas(config, k)
        shared = _shared_offset(config, dimension); external, beam = _external(config, dimension)
        mode = config.get("process_noise_mode")
        _require(mode in ("none", "tuned"), "Choose explicit process_noise_mode none or tuned for joint angular-kick least squares")
        piecewise = mode == "tuned"
        q = float(config.get("angular_variance_rad2_per_mm", np.nan)) if piecewise else 0.
        if piecewise:_require(np.isfinite(q) and q > 0., "Explicit positive angular variance [rad2/mm] required")
        model = "piecewise_linear_angular_kick_joint_least_squares" if piecewise else "straight_joint_generalized_least_squares"
        robust = config.get("robust", {}); kind = robust.get("kind", "none")
        _require(kind in ("none", "huber"), "Joint fit supports none or measurement-only Huber IRLS")
        cutoff = float(robust.get("cutoff", 2.5)); _require(cutoff > 0., "Positive Huber cutoff required")
        shared_n = int(shared > 0.)
        seed_lines = [np.linalg.lstsq(np.c_[np.ones(len(p)), p[:, 1]-p[0, 1]], p[:, axes], rcond=None)[0] for p in tracks]

        def layout(joint):
            names = (["vertex_" + s for s in (["u", "v"] if dimension == 2 else ["u", "v", "w"])] if joint else [])
            for i, p in enumerate(tracks):
                if piecewise:
                    if joint:names += ["track%d_vertex_slope%d" % (i, a) for a in range(m)]
                    for j in range(1 if joint else 0, len(p)):
                        names += ["track%d_node%d_transverse%d" % (i, j, a) for a in range(m)]
                else:
                    if not joint:names += ["track%d_first_transverse%d" % (i, a) for a in range(m)]
                    names += ["track%d_slope%d" % (i, a) for a in range(m)]
            if shared_n:names.append("shared_observation_w_offset_mm")
            return names

        def decode(parameters, joint):
            vertex = parameters[:dimension] if joint else None
            cursor = dimension if joint else 0
            points, local_slopes, vertex_slopes = [], [], []
            for i, p in enumerate(tracks):
                if piecewise:
                    if joint:
                        start = parameters[cursor:cursor+m]; cursor += m
                        rest = parameters[cursor:cursor+(len(p)-1)*m].reshape(-1, m); cursor += (len(p)-1)*m
                        transverse = np.vstack((vertex[axes] + start*(p[0, 1]-vertex[1]), rest))
                    else:
                        transverse = parameters[cursor:cursor+len(p)*m].reshape(-1, m); cursor += len(p)*m
                        start = (transverse[1]-transverse[0])/(p[1, 1]-p[0, 1])
                    segments = np.diff(transverse, axis=0)/np.diff(p[:, 1])[:, None]
                    slopes = np.vstack((segments, segments[-1]))
                else:
                    if joint:
                        slopes_value = parameters[cursor:cursor+m]; cursor += m
                        transverse = vertex[axes] + (p[:, 1]-vertex[1])[:, None]*slopes_value
                    else:
                        first = parameters[cursor:cursor+m]; cursor += m
                        slopes_value = parameters[cursor:cursor+m]; cursor += m
                        transverse = first + (p[:, 1]-references[i])[:, None]*slopes_value
                    slopes = np.tile(slopes_value, (len(p), 1)); start = slopes_value
                full = p.copy(); full[:, axes] = transverse
                points.append(full); local_slopes.append(slopes); vertex_slopes.append(start)
            offset = float(parameters[-1]) if shared_n else 0.
            return vertex, points, local_slopes, vertex_slopes, offset

        def residual(parameters, joint, weights, parts=False):
            vertex, points, slopes, starts, offset = decode(parameters, joint)
            measured, process, extra = [], [], []
            unweighted = []
            for i, p in enumerate(tracks):
                for j in range(len(p)):
                    r = _measurement_residual(p[j], points[i][j, axes], slopes[i][j], raw_cov[i][j], axes, offset)
                    measured.extend(r * math.sqrt(weights[i][j])); unweighted.append(np.linalg.norm(r))
                if piecewise:
                    segment_slopes = slopes[i][:-1]
                    lengths = np.linalg.norm(np.diff(points[i], axis=0), axis=1)
                    for j in range(1, len(segment_slopes)):
                        covariance = _angular_covariance((segment_slopes[j]+segment_slopes[j-1])/2., q*(lengths[j]+lengths[j-1])/2.)
                        process.extend(_white(segment_slopes[j]-segment_slopes[j-1], covariance))
                    if joint:
                        distance = float(np.linalg.norm(points[i][0]-vertex))
                        covariance = _angular_covariance((starts[i]+segment_slopes[0])/2., q*(distance+lengths[0])/2.)
                        process.extend(_white(segment_slopes[0]-starts[i], covariance))
            if joint:extra.extend(_extra_residual(vertex, references, sigma, external, beam, axes))
            if shared_n:extra.append(offset/shared)
            if parts:return measured, process, extra, unweighted
            return np.r_[measured, process, extra]

        def optimize(seed, joint):
            weights = [np.ones(len(p)) for p in tracks]
            fitted = None
            for _ in range(int(robust.get("iterations", 6)) if kind == "huber" else 1):
                fitted = least_squares(lambda x: residual(x, joint, weights), seed,
                    max_nfev=int(config.get("max_nfev", 300)), xtol=1e-10, ftol=1e-10, gtol=1e-10, x_scale="jac")
                if kind == "none":break
                norms = residual(fitted.x, joint, weights, True)[3]
                cursor = 0; changed = 0.
                for i, p in enumerate(tracks):
                    new = np.minimum(1., cutoff/np.maximum(norms[cursor:cursor+len(p)], 1e-300)); cursor += len(p)
                    changed = max(changed, float(np.max(abs(new-weights[i])))); weights[i] = new
                seed = fitted.x
                if changed < 1e-5:break
            # Align final solution/Jacobian with final IRLS weights.
            if kind != "none":
                fitted = least_squares(lambda x: residual(x, joint, weights), fitted.x,
                    max_nfev=int(config.get("max_nfev", 300)), x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10)
            covariance, rank, condition = _information_covariance(fitted.jac, config.get("condition_limit", 1e10))
            chunks = residual(fitted.x, joint, weights, True)
            measurement_rows = len(chunks[0])
            measurement_dof = float("nan")
            if np.isfinite(covariance).all():
                measured_jac = fitted.jac[:measurement_rows]
                leverage = float(np.sum((measured_jac @ covariance) * measured_jac))
                measurement_dof = max(1., measurement_rows - leverage)
            total_dof = max(1, len(fitted.fun) - rank)
            quality = dict(measurement_chi2=float(np.sum(np.square(chunks[0]))),
                           measurement_unweighted_chi2=float(np.sum(np.square(chunks[3]))),
                           process_chi2=float(np.sum(np.square(chunks[1]))),
                           support_and_external_chi2=float(np.sum(np.square(chunks[2]))),
                           solver_converged=bool(fitted.success), nfev=int(fitted.nfev), rank=rank,
                           parameter_count=len(fitted.x), scaled_jacobian_condition=condition,
                           measurement_weights=weights, measurement_count=sum(len(p) for p in tracks),
                           measurement_effective_dof_proxy=measurement_dof,
                           total_nominal_dof=total_dof,
                           measurement_reduced_chi2=float(np.sum(np.square(chunks[0]))/measurement_dof),
                           total_reduced_chi2=float(fitted.fun@fitted.fun/total_dof),
                           reduced_chi2_scope="numerical diagnostic with explicit covariance/process/robust model; not a calibrated probability")
            return fitted, covariance, quality

        free_seed = np.concatenate([p[:, axes].ravel() if piecewise else np.r_[line[0], line[1]]
                                    for p, line in zip(tracks, seed_lines)] + ([np.zeros(1)] if shared_n else []))
        free, free_cov, free_quality = optimize(free_seed, False)
        _, free_points, free_slopes, _, _ = decode(free.x, False)
        def local_states(parameters):
            _, points, slopes, _, _ = decode(parameters, False)
            return np.concatenate([np.r_[p[0, axes], s[0]] for p, s in zip(points, slopes)])
        state_jac = _finite_jacobian(local_states, free.x)
        state_cov = state_jac @ free_cov @ state_jac.T
        origins = np.stack([p[0] for p in free_points]); directions = _direction(np.stack([s[0] for s in free_slopes]), dimension)
        region_config = dict(config); region_config.pop("shared_w_offset", None)  # already in FULL free-state covariance
        region = vertex_region(origins, directions, state_cov, references, region_config, dimension)
        _require(region.get("candidate_point") is not None, region.get("message", "Vertex region unavailable"))
        vertex_seed = np.asarray(region["candidate_point"])
        joint_seed = [vertex_seed]
        for p, slopes in zip(free_points, free_slopes):
            joint_seed.append(slopes[0])
            if piecewise:joint_seed.append(p[1:, axes].ravel())
        if shared_n:joint_seed.append(np.zeros(1))
        fitted, covariance, quality = optimize(np.concatenate(joint_seed), True)
        vertex, points, slopes, vertex_slopes, offset = decode(fitted.x, True)
        # A full-rank local Hessian alone does not resolve a scan-censored vertex.
        scan = _scan(config, _reaction(config, dimension))
        joint_interval = [float("nan"), float("nan")]
        joint_open = [True, True]
        if np.isfinite(covariance).all() and covariance[1, 1] >= 0.:
            radius = math.sqrt(float(config.get("profile_delta_chi2", 3.841458820694124))*covariance[1, 1])
            joint_interval = [float(vertex[1]-radius), float(vertex[1]+radius)]
            joint_open = [joint_interval[0] <= scan[0], joint_interval[1] >= scan[1]]
        resolved = (quality["rank"] == len(fitted.x) and fitted.success and
                    region["status"] == "resolved" and not any(joint_open))
        gates = {"measurement_reduced_chi2": config.get("max_measurement_reduced_chi2"),
                 "total_reduced_chi2": config.get("max_total_reduced_chi2")}
        applied = any(limit is not None for limit in gates.values())
        failed_gates = []
        for key, limit in gates.items():
            if limit is not None:
                _require(np.isfinite(limit) and limit > 0., "Positive explicit quality gate required: max_"+key)
                if not np.isfinite(quality[key]) or quality[key] > limit:failed_gates.append(key)
        compatible = not failed_gates
        local = [_direction(s, dimension) for s in slopes]
        vertex_dirs = _direction(np.asarray(vertex_slopes), dimension)
        vertex_dir_cov = np.full((k*dimension, k*dimension), np.nan)
        if np.isfinite(covariance).all():
            direction_jac = _finite_jacobian(lambda x: _direction(np.asarray(decode(x, True)[3]), dimension).ravel(), fitted.x)
            vertex_dir_cov = direction_jac @ covariance @ direction_jac.T
        free_theta = np.degrees(np.arctan([s[0, 0] for s in free_slopes]))
        joint_theta = np.degrees(np.arctan([s[0, 0] for s in slopes]))
        result.update(status="incompatible_common_origin" if failed_gates else "ok" if resolved else "extended_region",
            success=resolved and compatible,
            vertex=vertex if resolved and compatible else None, candidate_vertex=vertex,
            vertex_covariance=covariance[:dimension, :dimension] if resolved and compatible else None,
            candidate_vertex_covariance=covariance[:dimension, :dimension],
            quality_gate_applied=applied, quality_gate_passed=compatible if applied else None,
            failed_quality_gates=failed_gates, configured_quality_gate_limits=gates,
            joint_covariance=covariance, joint_covariance_parameter_order=layout(True),
            joint_v_gaussian_interval_mm=joint_interval, joint_v_interval_open_sides=joint_open,
            free_covariance=free_cov, free_covariance_parameter_order=layout(False),
            covariance_classification="local Gauss-Newton covariance conditional on explicit measurement/process/robust model; uncalibrated",
            fitted_points=points, local_directions=local, free_fitted_points=free_points,
            free_local_directions=[_direction(s, dimension) for s in free_slopes],
            joint_track_directions=np.stack([d[0] for d in local]), free_track_directions=directions,
            vertex_track_directions=vertex_dirs, vertex_track_direction_joint_covariance=vertex_dir_cov,
            vertex_direction_scope="latent vertex-to-first-chord under this explicit model; unobserved eventwise deflection is not measured",
            vertex_free_theta_uv_deg=free_theta, vertex_joint_theta_uv_deg=joint_theta,
            quality=quality, free_quality=free_quality, region=region,
            same_measurement_process_model=True,
            free_comparison_note="same original measurements/covariance and angular diffusion; free origin/profiled first kick, no common-origin equality",
            external_prior_only_in_joint=external is not None or beam is not None,
            trajectory_model=model, full_Kalman_fit=False, shared_w_offset_fit_mm=offset,
            soft_first_layer_support_source=sigma_source, reaction_volume_not_clamped=True,
            independent_vertex_prior_from_current_tracks_used=False, dimension=dimension, **provenance)
        if _reaction(config, dimension) is not None:
            bounds = _reaction(config, dimension)
            inside_volume = bool(np.all(vertex >= bounds[0]) and np.all(vertex < bounds[1]))
            result["candidate_inside_reaction_volume"] = inside_volume
            if not inside_volume:
                result.update(status="outside_reaction_volume", success=False, vertex=None, vertex_covariance=None)
        return result
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        result["message"] = str(error)
        return result
