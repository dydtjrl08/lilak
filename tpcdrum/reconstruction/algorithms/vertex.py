"""Common-vertex DCA from upstream local tangents, in millimetres.

This is a geometrical estimator, not a Kalman smoother or a refit of entire
curved tracks. Each direction points from the beginning of the observed track
toward its downstream part. A valid vertex must lie within the configured
backward/forward extrapolation interval of *every* local tangent.

The sequence CNN predicts x-y angles only. For a 3D call, the caller must supply
the separately reconstructed z coordinate and z slope. This module never reads
MC positions, vertex truth, or learned model files. All dimensions must use mm;
do not substitute the finder's whitened diagnostic coordinates.
"""

from __future__ import annotations

import numpy as np


def _bound(value, n, name):
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must be finite, nonnegative, and scalar or length K")
    return values


def _pairwise(origins, directions):
    """Closest points on unbounded lines; these are diagnostic, not vertices."""
    result = []
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            a, b = origins[i], origins[j]
            u, v = directions[i], directions[j]
            r = a - b
            cosine = float(np.clip(u @ v, -1.0, 1.0))
            denominator = 1.0 - cosine * cosine
            parallel = denominator <= 1e-12
            item = {
                "tracks": [i, j],
                "opening_angle_deg": float(np.degrees(np.arccos(cosine))),
                "acute_line_angle_deg": float(np.degrees(np.arccos(abs(cosine)))),
                "parallel": parallel,
            }
            if parallel:
                # Distance is defined even when the closest pair is not unique.
                item.update(
                    dca_mm=float(np.linalg.norm(r - (r @ u) * u)),
                    closest_points_mm=None,
                    signed_extrapolation_mm=None,
                )
            else:
                ru, rv = float(r @ u), float(r @ v)
                si = (cosine * rv - ru) / denominator
                sj = (rv - cosine * ru) / denominator
                pi, pj = a + si * u, b + sj * v
                item.update(
                    dca_mm=float(np.linalg.norm(pi - pj)),
                    closest_points_mm=[pi.tolist(), pj.tolist()],
                    signed_extrapolation_mm=[float(si), float(sj)],
                )
            result.append(item)
    return result


def fit_common_vertex(
    origins_mm,
    directions,
    weights=None,
    *,
    max_backward_mm=500.0,
    max_forward_mm=0.0,
    condition_limit=1e6,
    max_dca_mm=None,
):
    """Minimize sum_i w_i ||(I-d_i d_i^T)(v-a_i)||^2 in 2D or 3D.

    Parameters
    ----------
    origins_mm, directions : arrays of shape (K, 2) or (K, 3)
        A measured point near the observed beginning and a downstream local
        tangent for each track. Directions need not be normalized.
    weights : optional positive length-K array
        Relative fit weights; default equal. They are not automatically
        interpreted as inverse measurement variances. No vertex uncertainty is
        inferred from them.
    max_backward_mm, max_forward_mm : nonnegative scalar or length-K array
        Allowed signed path distance from each origin along its unit tangent.
        Default: at most 500 mm backward and no forward extrapolation. This is
        a configurable geometry guard, not a calibrated detector parameter.
    condition_limit : float >= 1
        Reject a numerically ill-conditioned line bundle. Passing this guard
        does not guarantee good physical vertex resolution.
    max_dca_mm : optional finite nonnegative float
        Reject if any line's distance to the candidate exceeds this value.
        None disables this quality cut; it must be chosen using validation.

    Returns
    -------
    dict
        Strict JSON-compatible diagnostics. ``vertex_mm`` is populated only
        for valid fits. A finite unconstrained ``candidate_vertex_mm`` remains
        available after extrapolation/DCA rejection, but is never clamped to a
        track endpoint. Singular/ill-conditioned bundles have no candidate.
        Distances are geometrical; they are not fit uncertainties or chi2.
        A 2D result is only a projected vertex, not a 3D vertex.
    """
    origins = np.asarray(origins_mm, dtype=float)
    direction = np.asarray(directions, dtype=float)
    if origins.ndim != 2 or origins.shape[1] not in (2, 3):
        raise ValueError("origins_mm must have shape (K, 2) or (K, 3)")
    if direction.shape != origins.shape:
        raise ValueError("directions must have the same shape as origins_mm")
    if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(direction)):
        raise ValueError("origins_mm and directions must be finite")
    n, dim = origins.shape
    norms = np.linalg.norm(direction, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("each direction must have a finite nonzero length")
    direction = direction / norms[:, None]
    weight = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != (n,) or not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("weights must be a positive, finite length-K array")
    backward = _bound(max_backward_mm, n, "max_backward_mm")
    forward = _bound(max_forward_mm, n, "max_forward_mm")
    condition_limit = float(condition_limit)
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("condition_limit must be finite and at least 1")
    if max_dca_mm is not None:
        max_dca_mm = float(max_dca_mm)
        if not np.isfinite(max_dca_mm) or max_dca_mm < 0:
            raise ValueError("max_dca_mm must be None or finite and nonnegative")

    result = {
        "method": "local_tangent_common_dca",
        "dimension": dim,
        "n_tracks": n,
        "valid": False,
        "status": "insufficient_tracks",
        "flags": [],
        "vertex_mm": None,
        "candidate_vertex_mm": None,
        "origins_mm": origins.tolist(),
        "unit_directions": direction.tolist(),
        "weights": weight.tolist(),
        "max_backward_mm": backward.tolist(),
        "max_forward_mm": forward.tolist(),
        "max_dca_cut_mm": max_dca_mm,
        "condition_limit": condition_limit,
        "condition_number": None,
        "matrix_eigenvalues": None,
        "matrix_rank": None,
        "closest_points_mm": None,
        "signed_extrapolation_mm": None,
        "dca_mm": None,
        "weighted_rms_dca_mm": None,
        "max_dca_mm": None,
        "outside_bound_tracks": [],
        "over_dca_cut_tracks": [],
        "pairwise": _pairwise(origins, direction),
    }
    if n < 2:
        result["flags"].append("insufficient_tracks")
        return result

    # Normalize weights before building A to avoid numerical dependence on
    # their arbitrary overall scale. Center origins for translation stability.
    normalized_weight = weight / np.max(weight)
    normalized_weight /= np.sum(normalized_weight)
    center = np.sum(normalized_weight[:, None] * origins, axis=0)
    projectors = np.eye(dim)[None, :, :] - direction[:, :, None] * direction[:, None, :]
    matrix = np.sum(normalized_weight[:, None, None] * projectors, axis=0)
    rhs = np.sum(
        normalized_weight[:, None] * np.einsum("nij,nj->ni", projectors, origins - center),
        axis=0,
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_scale = float(np.max(eigenvalues))
    rank = int(np.count_nonzero(eigenvalues > eigen_scale * 1e-12))
    condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None
    result.update(
        matrix_eigenvalues=eigenvalues.tolist(),
        matrix_rank=rank,
        condition_number=condition if condition is not None and np.isfinite(condition) else None,
    )
    if rank < dim or condition is None or not np.isfinite(condition) or condition > condition_limit:
        result.update(status="parallel_or_ill_conditioned", flags=["parallel_or_ill_conditioned"])
        return result
    try:
        candidate = center + np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        result.update(status="singular_matrix", flags=["singular_matrix"])
        return result
    if not np.all(np.isfinite(candidate)):
        result.update(status="nonfinite_candidate", flags=["nonfinite_candidate"])
        return result

    signed_distance = np.einsum("ni,ni->n", candidate - origins, direction)
    closest = origins + signed_distance[:, None] * direction
    dca = np.linalg.norm(candidate - closest, axis=1)
    result.update(
        candidate_vertex_mm=candidate.tolist(),
        closest_points_mm=closest.tolist(),
        signed_extrapolation_mm=signed_distance.tolist(),
        dca_mm=dca.tolist(),
        weighted_rms_dca_mm=float(np.sqrt(np.sum(normalized_weight * dca * dca))),
        max_dca_mm=float(np.max(dca)),
    )
    # Floating point roundoff at s=0 should not turn a coincident vertex into a
    # rejected forward extrapolation. The tolerance is in millimetres.
    bound_tolerance_mm = 1e-7
    outside = (signed_distance < -backward - bound_tolerance_mm) | (
        signed_distance > forward + bound_tolerance_mm
    )
    result["outside_bound_tracks"] = np.flatnonzero(outside).astype(int).tolist()
    result["over_dca_cut_tracks"] = (
        np.flatnonzero(dca > max_dca_mm + bound_tolerance_mm).astype(int).tolist()
        if max_dca_mm is not None else []
    )
    if np.any(outside):
        result["flags"].append("outside_extrapolation_bounds")
    if result["over_dca_cut_tracks"]:
        result["flags"].append("large_dca")
    if result["flags"]:
        result["status"] = result["flags"][0]
    else:
        result.update(valid=True, status="ok", vertex_mm=candidate.tolist())
    return result


def vertex_from_paths(paths_mm, tangent_segments=1, **fit_kwargs):
    """Use each observed path's first local chord; never globally refit a path.

    Paths must have a shared dimension (2 or 3), finite coordinates in mm, and
    strictly increasing layer-y (coordinate index 1). ``tangent_segments=1``
    uses points 0 and 1. Larger values explicitly average over a longer initial
    chord. A path with fewer requested segments raises ValueError rather than
    silently changing the vertex definition. For explicit CNN initial-head
    directions, call ``fit_common_vertex`` directly instead.
    """
    if not isinstance(tangent_segments, (int, np.integer)) or isinstance(tangent_segments, bool) or tangent_segments < 1:
        raise ValueError("tangent_segments must be a positive integer")
    paths = [np.asarray(path, dtype=float) for path in paths_mm]
    if not paths:
        raise ValueError("paths_mm must contain at least one path")
    dim = paths[0].shape[1] if paths[0].ndim == 2 else None
    if dim not in (2, 3):
        raise ValueError("each path must have shape (N, 2) or (N, 3)")
    for path in paths:
        if path.ndim != 2 or path.shape[1] != dim or len(path) <= tangent_segments:
            raise ValueError("paths must share a dimension and contain the requested initial chord")
        if not np.all(np.isfinite(path)) or np.any(np.diff(path[:, 1]) <= 0):
            raise ValueError("paths must be finite and ordered in strictly increasing layer-y")
    origins = np.stack([path[0] for path in paths])
    directions = np.stack([path[tangent_segments] - path[0] for path in paths])
    result = fit_common_vertex(origins, directions, **fit_kwargs)
    result["tangent_source"] = "first_path_chord"
    result["tangent_segments"] = int(tangent_segments)
    result["tangent_span_mm"] = np.linalg.norm(directions, axis=1).tolist()
    result["tangent_layer_y_span_mm"] = directions[:, 1].tolist()
    return result
