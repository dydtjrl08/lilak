"""Conditional energy-aware refitting with explicit downstream invalidation.

The energy prior is at the FIRST OBSERVED plane. It comes from the previous
energy stage and depends on that previous trajectory. This iterative conditional
update is not an independent joint Bayesian solution. New trajectories must go
through vertex -> Si association -> material request/trace/import -> energy
again. Old traced segment lengths are NEVER applied to the altered trajectory.
"""
from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

from .algorithms.smoother import fit_track
from .algorithms.track_fit import _empty
from .physics_stages import (_tables, _decode, _energy_empty, _request_empty,
                             _material_empty, _association_empty)
from .kinematics_stage import _empty as _empty_kinematics


def _k(record):
    for name in ("fit_ok", "track_first_point_energy_mev", "components"):
        if name in record:
            return len(record[name])
    return 0


def _downstream_defaults(record, k):
    n = len(record.get("si_edep_mev", []))
    association = _association_empty(k, n)
    association.pop("si_geometry_json_utf8")  # Static calibration remains usable.
    association.pop("si_geometry_observed_index")  # Name alignment is not a track assignment.
    result = dict(association, **_request_empty(k), **_energy_empty(k))
    result.update({key: value for key, value in _material_empty().items()
                   if key.startswith("material_path")})
    result.update(_empty_kinematics())
    if 'vertex_direction_valid' in record:
        result['vertex_direction_valid']=np.zeros(k,dtype=np.uint8)
    if 'track_vertex_directions_uvw' in record:
        result['track_vertex_directions_uvw']=np.full((k,3),np.nan)
    result.update(vertex_valid=np.int64(0), vertex_uvw_mm=np.full(3, np.nan),
                  vertex_max_dca_mm=np.float64(np.nan), vertex_condition_number=np.float64(np.nan),
                  kinematics_ok=np.int64(0), invariant_mass_mev=np.float64(np.nan),
                  excitation_mev=np.float64(np.nan), com_opening_angles_deg=np.full(3, np.nan),
                  dalitz_xy=np.full(2, np.nan), alpha_four_vectors_mev=np.full((3, 4), np.nan),
                  com_four_vectors_mev=np.full((3, 4), np.nan))
    return result


def _snapshot(record, name, current):
    # Keep the first pre-refit baseline across any later conditional iterations.
    return np.asarray(record.get("initial_" + name, current)).copy()


def _starting_updates(record, k, requested):
    geometry = {key: np.asarray(record.get(key, value)).copy() for key, value in _empty(k).items()}
    downstream = {key: np.asarray(record.get(key, value)).copy()
                  for key, value in _downstream_defaults(record, k).items()}
    result = dict(geometry, **downstream)
    for key, value in result.copy().items():
        result["initial_" + key] = _snapshot(record, key, value)
    result.update(refit_requested=np.int64(requested), refit_attempted=np.int64(0),
                  refit_attempted_track=np.zeros(k, dtype=np.uint8),
                  refit_ok=np.zeros(k, dtype=np.uint8),
                  refit_geometry_changed=np.zeros(k, dtype=np.uint8),
                  refit_max_point_change_mm=np.full(k, np.nan),
                  refit_prior_energy_mev=np.full(k, np.nan),
                  refit_prior_energy_sigma_mev=np.full(k, np.nan),
                  refit_input_reference_v_mm=np.full(k, np.nan),
                  refit_requires_downstream_recompute=np.int64(0),
                  refit_revision=np.int64(record.get("refit_revision", 0)))
    return result


def _table(record, options):
    selected = options.get("material_name")
    explicit = options.get("stopping_table")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError("refitting.stopping_table must be an explicit numeric mapping")
        explicit = copy.deepcopy(dict(explicit))
        selected = selected or explicit.get("material")
        if not selected or explicit.get("material") != selected:
            raise ValueError("Explicit stopping table and refitting.material_name must agree")
        return explicit, str(selected), "explicit_refitting_stopping_table"
    if not selected:
        raise ValueError("Explicit refitting.material_name is required to select an embedded table")
    # The ROOT payload is authoritative here; never re-open an old upstream JSON
    # merely because config.energy.stopping_tables still names that vanished file.
    tables, _, _ = _tables(record, {})
    if selected not in tables:
        raise ValueError("Selected material is absent from embedded stopping payload: " + str(selected))
    return tables[selected].mapping(), str(selected), "embedded_ROOT_stopping_payload"


def _uniform_material(record, track, observed_count, material, options):
    if bool(record.get("material_paths_embedded_valid", False)):
        metadata = _decode(record["stopping_metadata_json_utf8"])
        names = [item["material"] for item in metadata["table_metadata"]]
        tracks = np.asarray(record["material_path_track"], dtype=int)
        segments = np.asarray(record["material_path_segment_id"], dtype=int)
        indices = np.asarray(record["material_path_table_index"], dtype=int)
        lengths = np.asarray(record["material_path_length_mm"], dtype=float)
        if (tracks.ndim != 1 or segments.shape != tracks.shape or indices.shape != tracks.shape or
                lengths.shape != tracks.shape or not np.all(np.isfinite(lengths)) or
                np.any(lengths < 0.) or np.any(indices < 0) or np.any(indices >= len(names))):
            raise ValueError("Invalid embedded material proof for input trajectory")
        boundary = int(np.asarray(record["material_request_first_segment_id"])[track])
        for segment in range(boundary, boundary + observed_count - 1):
            mask = (tracks == track) & (segments == segment) & (lengths > 0.)
            if not np.any(mask):
                raise ValueError("Missing native material pieces between observed planes")
            if any(names[index] != material for index in indices[mask]):
                raise ValueError("Observed input path crosses multiple/different materials; homogeneous refitter cannot omit them")
        return "single_material_on_PREVIOUS_traced_observed_segments_only"
    if options.get("homogeneous_material_between_points") is not True:
        raise ValueError("No traced homogeneous inter-plane material proof; explicitly declare homogeneous_material_between_points=true as a model assumption")
    return "explicit_homogeneous_model_assumption_NOT_native_material_validation"


def _configuration(options, track, k, sigma):
    result = copy.deepcopy(dict(options))
    overrides = result.pop("per_track", None)
    if overrides is not None:
        if isinstance(overrides, Mapping):
            override = overrides.get(str(track), overrides.get(track, {}))
        else:
            if len(overrides) != k:
                raise ValueError("refitting.per_track must have K entries")
            override = overrides[track]
        result.update(copy.deepcopy(dict(override)))
    for forbidden in ("energy_mev", "energy_first_point_mev", "vertex_energy_mev"):
        if forbidden in result:
            raise ValueError("Conditional refit consumes energy-stage first-point priors, not config." + forbidden)
    if result.get("method") not in ("ekf_rts", "ukf_rts", "daf_ekf", "daf_ukf"):
        raise ValueError("An explicit energy-aware EKF/UKF refitting method is required")
    for field in ("initial_covariance", "straggling_variance_mev2_per_mm", "process_noise_mode"):
        if result.get(field) is None:
            raise ValueError("Missing explicit refitting." + field)
    if "energy_sigma_mev" in result:
        declared = np.asarray(result["energy_sigma_mev"], dtype=float)
        sigma = float(declared[track] if declared.shape == (k,) else declared)
    if sigma is None or not np.isfinite(sigma) or sigma < 0.:
        raise ValueError("Missing finite first-point energy uncertainty (record or explicit refitting.energy_sigma_mev)")
    result["energy_sigma_mev"] = float(sigma)
    covariance = np.asarray(result["initial_covariance"], dtype=float)
    if covariance.ndim == 3 and covariance.shape[0] == k:
        covariance = covariance[track]
    if covariance.shape == (4, 4):
        full = np.zeros((5, 5)); full[:4, :4] = covariance; full[4, 4] = sigma ** 2
        covariance = full
    if covariance.shape != (5, 5):
        raise ValueError("Refitting initial_covariance must be explicit 4x4 geometry or 5x5 full covariance")
    result["initial_covariance"] = covariance
    if "initial_state" in result:
        initial = np.asarray(result["initial_state"], dtype=float)
        if initial.ndim == 2 and initial.shape[0] == k:
            initial = initial[track]
        result["initial_state"] = initial
    if result.get("energy_reference", "first_point") != "first_point":
        raise ValueError("Refitting energy reference must be first_point, never vertex or Si")
    result["energy_reference"] = "first_point"
    result["coordinate_system"] = "physical_uvw_mm"
    return result


def _changed(record, track, layers, result):
    previous = np.asarray(record["fit_points_uvw_mm"], dtype=float)[track, layers]
    current = np.asarray(result["fitted_points_uvw_mm"])
    difference = np.linalg.norm(current - previous, axis=1)
    max_change = float(np.max(difference)) if np.all(np.isfinite(difference)) else float("nan")
    old = np.asarray(record["fit_states"], dtype=float)[track, layers, :4]
    new = np.asarray(result["states"])[:, :4]
    direction_changed = ("track_directions_uvw" not in record or not np.allclose(
        np.asarray(record["track_directions_uvw"])[track], result["initial_downstream_direction"],
        rtol=0., atol=1e-8, equal_nan=False))
    return (direction_changed or not np.allclose(old, new, rtol=0., atol=1e-8, equal_nan=False) or
            not np.allclose(previous, current, rtol=0., atol=1e-8, equal_nan=False)), max_change


def refitting_event(record, diagnostic, config):
    """Standalone ROOT-record handler; returns numeric updates and merged diagnostics.

    Uses embedded OBSERVED point measurements, not a previous CNN/Kalman fit as
    a second measurement. On any successful replacement all downstream physics
    is invalidated event-wide, even if coordinate changes are below tolerance.
    Material property tables and original Si observations remain untouched.
    """
    options = dict(config.get("refitting", {}))
    k = _k(record)
    values = _starting_updates(record, k, options.get("enabled") is True)
    result = dict(status="disabled", tracks=[], physical_3d=False,
                  prior_reference="first observed plane, NOT production vertex",
                  conditional_iteration=True, correlated_prior_and_geometry_not_jointly_solved=True,
                  cross_track_energy_prior_covariance_propagated=False,
                  new_material_path_validated=False, physics_accuracy_evaluated=False,
                  physical_uncertainty_validated=False)
    merged = dict(diagnostic)
    if options.get("enabled") is not True:
        return values, dict(merged, refitting=result)
    try:
        if not diagnostic.get("fitting", {}).get("physical_3d", False):
            raise ValueError("Missing calibrated physical-3D input fitting provenance")
        points = np.asarray(record["fit_measurement_points_uvw_mm"], dtype=float)
        covariance = np.asarray(record["fit_measurement_covariance_mm2"], dtype=float)
        valid = np.asarray(record["fit_valid_layer"], dtype=bool)
        fit_ok = np.asarray(record["fit_ok"], dtype=bool)
        energy_ok = np.asarray(record["energy_ok"], dtype=bool)
        energy = np.asarray(record["track_first_point_energy_mev"], dtype=float)
        sigma = np.asarray(record.get("track_first_point_energy_sigma_mev", np.full(k, np.nan)), dtype=float)
        if (points.shape != (k, 12, 3) or covariance.shape != (k, 12, 3, 3) or
                valid.shape != (k, 12) or fit_ok.shape != (k,) or energy_ok.shape != (k,) or
                energy.shape != (k,) or sigma.shape != (k,)):
            raise ValueError("Invalid embedded observation/energy-prior shapes")
        table, material, table_source = _table(record, options)
        result.update(stopping_table_source=table_source, material_name=material)
        minimum = int(options.get("minimum_layers", 3))
        if minimum < 3 or minimum > 12:
            raise ValueError("refitting.minimum_layers must be 3..12")
        result["status"] = "no_valid_tracks"
        for track in range(k):
            info = dict(track=track, status="missing_inputs")
            try:
                if not fit_ok[track] or not energy_ok[track] or not np.isfinite(energy[track]) or energy[track] <= 0.:
                    raise ValueError("Valid current input fit and energy-stage first-point prior are required")
                layers = np.flatnonzero(valid[track])
                if len(layers) < minimum or not np.isfinite(points[track, layers]).all() or not np.isfinite(covariance[track, layers]).all():
                    raise ValueError("Insufficient finite embedded observed point measurements/covariances")
                layers = layers[np.argsort(points[track, layers, 1], kind="stable")]
                reference = points[track, layers[0], 1]
                old_points = np.asarray(record["fit_points_uvw_mm"], dtype=float)
                old_reference = old_points[track, layers[0], 1]
                if "track_first_fit_point_uvw_mm" in record:
                    old_reference = np.asarray(record["track_first_fit_point_uvw_mm"])[track, 1]
                if not np.isfinite(old_reference) or abs(reference - old_reference) > 1e-8:
                    raise ValueError("First-point energy and observed measurement depth references disagree")
                material_scope = _uniform_material(record, track, len(layers), material, options)
                settings = _configuration(options, track, k, sigma[track])
                initial = settings.get("initial_state")
                if initial is not None and np.asarray(initial).shape == (4,):
                    settings["initial_state"] = np.r_[initial, energy[track]]
                values["refit_prior_energy_mev"][track] = energy[track]
                values["refit_prior_energy_sigma_mev"][track] = settings["energy_sigma_mev"]
                values["refit_input_reference_v_mm"][track] = reference
                values["refit_attempted_track"][track] = 1
                values["refit_attempted"] = np.int64(1)
                fitted = fit_track(points[track, layers], covariance[track, layers], settings,
                                   energy_mev=energy[track], stopping_table=table)
                info.update(fitted)
                info["input_material_scope"] = material_scope
                if not fitted.get("success", False):
                    result["tracks"].append(info)
                    continue
                layers = layers[np.asarray(fitted["input_order"], dtype=int)]
                changed, shift = _changed(record, track, layers, fitted)
                for field, key in (("fit_states", "states"), ("fit_covariances", "covariances"),
                                   ("fit_points_uvw_mm", "fitted_points_uvw_mm"),
                                   ("fit_innovation_pulls", "innovation_pulls"), ("fit_daf_weights", "daf_weights")):
                    values[field][track, layers] = fitted[key]
                values["track_origins_uvw_mm"][track] = fitted["fitted_points_uvw_mm"][0]
                values["track_directions_uvw"][track] = fitted["initial_downstream_direction"]
                values["track_direction_covariances"][track] = fitted["initial_direction_covariance"]
                values["fit_reference_v_mm"][track] = fitted["initial_reference_v_mm"]
                values["fit_state_dimension"][track] = 5
                values["fit_energy_first_point_mev"][track] = fitted["states"][0, 4]
                values["fit_ok"][track] = 1
                values["refit_ok"][track] = 1
                values["refit_geometry_changed"][track] = changed
                values["refit_max_point_change_mm"][track] = shift
            except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
                info.update(status="missing_or_invalid_calibration", message=str(error))
            result["tracks"].append(info)
        successful = int(values["refit_ok"].sum())
        result.update(successful_tracks=successful, physical_3d=True,
                      status="ok" if k and successful == k else "partial_refit" if successful else "missing_or_invalid_calibration")
        if successful:
            values.update(_downstream_defaults(record, k))
            values["refit_requires_downstream_recompute"] = np.int64(1)
            values["refit_revision"] = np.int64(int(record.get("refit_revision", 0)) + 1)
            result.update(downstream_invalidated=True,
                          required_next_stages=["vertex", "association", "material_request", "native_material_trace", "material_import", "energy", "kinematics"],
                          invalidation_policy="Any successful refit replacement invalidates all downstream results, even at negligible coordinate change")
            for section in ("fitting", "vertex", "association", "material_request", "material_import", "energy", "kinematics"):
                if "initial_" + section not in merged and section in merged:
                    merged["initial_" + section] = copy.deepcopy(merged[section])
                if section != "fitting":
                    merged[section] = dict(status="invalidated_by_refit", valid=False,
                                          requires_recompute=True, refit_revision=int(values["refit_revision"]))
            merged["fitting"] = dict(status=result["status"], physical_3d=True,
                                    method="conditional_energy_aware_refit", tracks=result["tracks"],
                                    physical_uncertainty_validated=False, physics_accuracy_evaluated=False)
        return values, dict(merged, refitting=result)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        result.update(status="missing_or_invalid_calibration", message=str(error))
        return values, dict(merged, refitting=result)
