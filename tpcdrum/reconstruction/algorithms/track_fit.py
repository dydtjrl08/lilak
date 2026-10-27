"""Observation-to-Kalman bridge with embedded, fixed-shape stage outputs.

Charge-cloud widths below are SAMPLING/SPREAD PROXIES, not calibrated errors on
the centroid. ADC is not an electron count: no invented 1/sqrt(ADC) precision is
used. Conditional fitting does not validate upstream component separation.
"""
from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

from ..preprocessing import PAD_U_MM, PAD_WIDTH_MM, LAYER_V_MM, drift_coordinate
from .smoother import fit_track


def _empty(k):
    def nan(shape):
        return np.full(shape, np.nan, dtype=np.float64)
    return dict(fit_states=nan((k, 12, 5)), fit_covariances=nan((k, 12, 5, 5)),
                fit_points_uvw_mm=nan((k, 12, 3)),
                fit_valid_layer=np.zeros((k, 12), dtype=np.uint8),
                fit_ok=np.zeros(k, dtype=np.uint8),
                track_origins_uvw_mm=nan((k, 3)), track_directions_uvw=nan((k, 3)),
                track_direction_covariances=nan((k, 3, 3)),
                fit_innovation_pulls=nan((k, 12, 2)), fit_daf_weights=np.zeros((k, 12)),
                fit_energy_first_point_mev=nan(k),
                fit_measurement_points_uvw_mm=nan((k, 12, 3)),
                fit_measurement_covariance_mm2=nan((k, 12, 3, 3)),
                fit_layer_charge=np.zeros((k, 12), dtype=np.float64),
                fit_theta_pad_2d_ols_deg=nan(k),
                fit_reference_v_mm=nan(k), fit_state_dimension=np.zeros(k, dtype=np.int64))


def _drift(calibration, record):
    """Validate explicit charge-mean conversion before constructing physical w."""
    calibration = dict(calibration or {})
    required = ["calibrated_for_charge_mean", "velocity_mm_per_ns", "t0_tb",
                "reference_w_mm", "sign", "tb_ns"]
    missing = [key for key in required if key not in calibration]
    if calibration.get("calibrated_for_charge_mean") is not True:
        missing.append("validated_charge_mean_timing_conversion")
    if missing:
        return None, dict(status="missing_calibration", missing_inputs=sorted(set(missing)),
                          message="Physical w requires explicit charge-mean timing calibration")
    try:
        drift_coordinate(np.array([0., 1.]), calibration)
        if "tb_ns" in record and not np.isclose(float(np.asarray(record["tb_ns"])),
                                                 float(calibration["tb_ns"]), rtol=0., atol=1e-9):
            raise ValueError("Record TB duration differs from drift calibration")
        if calibration.get("pad_delay_npz"):
            raise ValueError("Embed pad_delay_tb values in calibration; this bridge does not read a hidden NPZ")
        delay = np.asarray(calibration.get("pad_delay_tb", np.zeros((12, 64))), dtype=float)
        if delay.shape == (64,):
            delay = np.broadcast_to(delay, (12, 64)).copy()
        if delay.shape != (12, 64) or not np.all(np.isfinite(delay)):
            raise ValueError("pad_delay_tb must have shape (12,64) or (64,)")
        calibration["_delay_tb"] = delay
        return calibration, None
    except (ValueError, TypeError, OverflowError) as error:
        return None, dict(status="invalid_calibration", missing_inputs=[], message=str(error))


def _waveform_measurements(record, fitting, calibration):
    components = np.asarray(record["components"], dtype=float)
    if (components.ndim != 4 or components.shape[1:] != (12, 64, 512) or
            not np.all(np.isfinite(components)) or np.any(components < 0.)):
        raise ValueError("components must be finite nonnegative (K,12,64,512) observed partitions")
    k = components.shape[0]
    positions = np.full((k, 12, 3), np.nan)
    covariance = np.full((k, 12, 3, 3), np.nan)
    charge = components.sum(axis=(2, 3))
    minimum_charge = float(fitting.get("minimum_layer_charge_adc_tb", 0.))
    minimum_peak = float(fitting.get("minimum_layer_peak_adc", 0.))
    if not np.isfinite([minimum_charge, minimum_peak]).all() or min(minimum_charge, minimum_peak) < 0.:
        raise ValueError("Layer charge/peak gates must be finite and nonnegative")
    valid = (charge > minimum_charge) & (components.max(axis=(2, 3)) > minimum_peak)
    t = np.arange(512., dtype=float)[None, :]
    width_u = PAD_WIDTH_MM ** 2 / 12.
    for track in range(k):
        for layer in np.flatnonzero(valid[track]):
            q = components[track, layer]
            total = charge[track, layer]
            pad_weights = q.sum(axis=1) / total
            mean_u = float(pad_weights @ PAD_U_MM)
            positions[track, layer, :2] = [mean_u, LAYER_V_MM[layer]]
            if calibration is None:
                continue
            corrected_tb = t - calibration["_delay_tb"][layer, :, None]
            mean_tb = float(np.sum(q * corrected_tb) / total)
            scale = float(calibration["sign"]) * float(calibration["tb_ns"]) * float(calibration["velocity_mm_per_ns"])
            positions[track, layer, 2] = float(drift_coordinate(mean_tb, calibration))
            du = PAD_U_MM - mean_u
            dt = corrected_tb - mean_tb
            uu = float(pad_weights @ (du ** 2 + width_u))
            tt = float(np.sum(q * dt ** 2) / total) + 1. / 12.
            ut = float(np.sum(q * du[:, None] * dt) / total)
            covariance[track, layer] = [[uu, 0., scale * ut],
                                        [0., 12. ** 2 / 12., 0.],
                                        [scale * ut, 0., scale ** 2 * tt]]
    return positions, covariance, charge, valid


def _point_measurements(record, calibration, fitting):
    points = np.asarray(record["points_uvw_mm"], dtype=float)
    covariance = np.asarray(record["point_covariance_mm2"], dtype=float)
    charge = np.asarray(record["point_charge"], dtype=float)
    layer = np.asarray(record["point_layer"])
    fractions = np.asarray(record["point_fractions"], dtype=float)
    n = len(points)
    if (points.shape != (n, 3) or covariance.shape != (n, 3, 3) or charge.shape != (n,) or
            layer.shape != (n,) or fractions.ndim != 2 or fractions.shape[0] != n or
            fractions.shape[1] < 2):
        raise ValueError("Invalid space-point/group covariance/charge/layer/fraction shapes")
    if (not np.all(np.isfinite(points[:, :2])) or not np.all(np.isfinite(charge)) or
            np.any(charge < 0.) or not np.all(np.isfinite(fractions)) or
            np.any(fractions < 0.) or np.any(fractions > 1.) or
            not np.allclose(fractions.sum(axis=1), 1., atol=1e-5, rtol=0.) or
            not np.all(np.isfinite(layer)) or not np.all(layer == np.floor(layer)) or
            np.any(layer < 0) or np.any(layer > 11)):
        raise ValueError("Invalid observation space-point weights or layer IDs")
    k = fractions.shape[1] - 1
    positions = np.full((k, 12, 3), np.nan)
    cov = np.full((k, 12, 3, 3), np.nan)
    total = np.zeros((k, 12)); valid = np.zeros((k, 12), dtype=bool)
    physical = calibration is not None and bool(np.asarray(record.get("physical_3d", False)).item())
    if physical and (not np.all(np.isfinite(points)) or not np.all(np.isfinite(covariance))):
        raise ValueError("Physical space-point coordinates/covariances contain nonfinite entries")
    if physical:
        for matrix in covariance:
            if (not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-8) or
                    np.linalg.eigvalsh(matrix).min() < -1e-10):
                raise ValueError("Invalid individual space-point covariance")
        if np.any(calibration["_delay_tb"]):
            # Recompute rather than double-subtracting an unknown upstream pad
            # correction. The previous stage embeds the observed timing fields.
            tb = np.asarray(record.get("point_tb"), dtype=float)
            pad = np.asarray(record.get("point_pad"))
            if (tb.shape != (n,) or pad.shape != (n,) or not np.all(np.isfinite(tb)) or
                    not np.all(np.isfinite(pad)) or np.any(pad != np.floor(pad)) or
                    np.any(pad < 0) or np.any(pad > 63)):
                raise ValueError("Nonzero pad delays on space points require embedded point_tb and point_pad")
            points = points.copy()
            points[:, 2] = drift_coordinate(tb - calibration["_delay_tb"][layer.astype(int), pad.astype(int)], calibration)
    minimum_charge = float(fitting.get("minimum_layer_charge_adc_tb", 0.))
    if not np.isfinite(minimum_charge) or minimum_charge < 0.:
        raise ValueError("minimum_layer_charge_adc_tb must be finite and nonnegative")
    for track in range(k):
        for depth in range(12):
            ids = np.flatnonzero((layer == depth) & (charge * fractions[:, track] > 0.))
            if not len(ids):
                continue
            weights = charge[ids] * fractions[ids, track]
            total[track, depth] = weights.sum()
            if total[track, depth] <= minimum_charge:
                continue
            alpha = weights / total[track, depth]
            positions[track, depth, :2] = alpha @ points[ids, :2]
            valid[track, depth] = True
            if not physical:
                continue
            positions[track, depth] = alpha @ points[ids]
            centered = points[ids] - positions[track, depth]
            # Law of total spread, not the variance-of-the-mean formula. Soft
            # assignments and intra-point pulse widths are not independent hits.
            cov[track, depth] = (np.einsum("i,ijk->jk", alpha, covariance[ids]) +
                                 (centered.T * alpha) @ centered)
    return positions, cov, total, valid, physical


def _track_config(fitting, track, k, record):
    fitting = copy.deepcopy(dict(fitting))
    overrides = fitting.pop("per_track", None)
    if overrides is not None:
        if isinstance(overrides, Mapping):
            override = overrides.get(str(track), overrides.get(track, {}))
        else:
            if len(overrides) != k:
                raise ValueError("fitting.per_track must have K entries")
            override = overrides[track]
        fitting.update(copy.deepcopy(dict(override)))
    for key, ndim in (("initial_state", 2), ("initial_covariance", 3)):
        if key in fitting:
            array = np.asarray(fitting[key])
            if array.ndim == ndim and array.shape[0] == k and array.shape[1] in (4, 5):
                fitting[key] = array[track].tolist()
    if "energy_sigma_mev" in fitting and np.asarray(fitting["energy_sigma_mev"]).shape == (k,):
        fitting["energy_sigma_mev"] = float(np.asarray(fitting["energy_sigma_mev"])[track])
    prior = fitting.pop("energy_first_point_mev", record.get("energy_first_point_prior_mev"))
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        if prior.ndim == 0:
            prior = float(prior)
        elif prior.shape == (k,):
            prior = float(prior[track])
        else:
            raise ValueError("energy_first_point_mev prior must be scalar or K entries")
    stopping = fitting.pop("stopping_table", None)
    return fitting, prior, stopping


def fit_event(record, config):
    """Return (numeric_updates, diagnostics) without ROOT, cache or truth access.

    Direct route: components are measured-charge partitions, not predicted clean
    templates. Space-point route: point_fractions has K track columns followed
    by background. Both routes retain one weighted cloud measurement per layer.
    ``fit_valid_layer`` means usable physical measurement support; ``fit_ok``
    separately reports a successful conditional track fit. Missing layers stay
    NaN and are never interpolated into observations.
    """
    try:
        if not isinstance(record, Mapping) or not isinstance(config, Mapping):
            raise ValueError("record and config must be mappings")
        route = config.get("route", "spacepoint" if "point_fractions" in record else "direct")
        space = route in ("spacepoint", "space_points", "space_point")
        k = int(config.get("finding", {}).get("k", 0))
        if space and "point_fractions" in record and np.ndim(record["point_fractions"]) == 2:
            k = max(0, np.shape(record["point_fractions"])[1] - 1)
        elif not space and "components" in record and np.ndim(record["components"]) == 4:
            k = np.shape(record["components"])[0]
        if k < 0:
            raise ValueError("Negative track multiplicity")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return _empty(0), dict(status="invalid_input", message=str(error), tracks=[],
                               physics_accuracy_evaluated=False, physical_3d=False)
    updates = _empty(k)
    diagnostics = dict(status="invalid_input", route="spacepoint" if space else "direct_waveform",
                       tracks=[], physical_3d=False, physics_accuracy_evaluated=False,
                       covariance_model="charge-cloud/pulse-spread and sampling proxy; NOT a calibrated centroid resolution",
                       component_assignment_uncertainty_included=False,
                       conditional_fit_does_not_validate_track_separation=True,
                       first_direction_definition="downstream tangent at first observed plane, not MC production direction")
    try:
        if route not in ("spacepoint", "space_points", "space_point", "direct", "waveform", "direct_waveform"):
            raise ValueError("Unknown fitting representation route")
        fitting = dict(config.get("fitting", {}))
        min_layers = int(fitting.get("minimum_layers", 4))
        if min_layers < 3 or min_layers > 12:
            raise ValueError("minimum_layers must be between 3 and 12")
        calibration, drift_error = _drift(config.get("drift"), record)
        if space:
            measurements, covariance, charge, valid, physical = _point_measurements(record, calibration, fitting)
            if calibration is not None and not physical:
                drift_error = dict(status="missing_calibration", missing_inputs=["physical_3d_point_provenance"],
                                   message="Space-point record is not marked as calibrated physical 3D")
        else:
            measurements, covariance, charge, valid = _waveform_measurements(record, fitting, calibration)
            physical = calibration is not None
        updates["fit_layer_charge"] = charge
        diagnostics["layer_selection"] = dict(minimum_layers=min_layers,
                                             minimum_layer_charge_adc_tb=fitting.get("minimum_layer_charge_adc_tb", 0.),
                                             minimum_layer_peak_adc=fitting.get("minimum_layer_peak_adc", 0.))
        for track in range(k):
            ids = np.flatnonzero(valid[track])
            info = dict(track=track, observed_layers=ids.tolist(), observed_layer_count=len(ids),
                        physical_uncertainty_validated=False)
            if len(ids) >= 3:
                a = np.column_stack((np.ones(len(ids)), measurements[track, ids, 1]))
                if np.linalg.matrix_rank(a) == 2:
                    slope = np.linalg.lstsq(a, measurements[track, ids, 0], rcond=None)[0][1]
                    updates["fit_theta_pad_2d_ols_deg"][track] = np.degrees(np.arctan(slope))
                    info["projected_theta_note"] = "unweighted u-v OLS diagnostic only; does not determine physical 3D direction"
            if drift_error is not None:
                info.update(drift_error)
                diagnostics["tracks"].append(info)
                continue
            updates["fit_measurement_points_uvw_mm"][track, ids] = measurements[track, ids]
            updates["fit_measurement_covariance_mm2"][track, ids] = covariance[track, ids]
            updates["fit_valid_layer"][track, ids] = 1
            if len(ids) < min_layers:
                info.update(status="insufficient_layers", message="Too few observed distinct layers")
                diagnostics["tracks"].append(info)
                continue
            try:
                track_config, energy, stopping = _track_config(fitting, track, k, record)
            except (ValueError, TypeError, KeyError) as error:
                info.update(status="invalid_input", message=str(error))
                diagnostics["tracks"].append(info)
                continue
            result = fit_track(measurements[track, ids], covariance[track, ids], track_config,
                               energy_mev=energy, stopping_table=stopping)
            # Full smoother diagnostics contain only observation/current-stage
            # calibration inputs; preserve them in the stage diagnostics JSON.
            info.update(result)
            if result.get("success"):
                order = np.asarray(result["input_order"], dtype=int)
                layers = ids[order]
                updates["fit_states"][track, layers] = result["states"]
                updates["fit_covariances"][track, layers] = result["covariances"]
                updates["fit_points_uvw_mm"][track, layers] = result["fitted_points_uvw_mm"]
                updates["fit_innovation_pulls"][track, layers] = result["innovation_pulls"]
                updates["fit_daf_weights"][track, layers] = result["daf_weights"]
                updates["track_origins_uvw_mm"][track] = result["fitted_points_uvw_mm"][0]
                updates["track_directions_uvw"][track] = result["initial_downstream_direction"]
                updates["track_direction_covariances"][track] = result["initial_direction_covariance"]
                updates["fit_reference_v_mm"][track] = result["initial_reference_v_mm"]
                updates["fit_energy_first_point_mev"][track] = result["states"][0, 4]
                updates["fit_state_dimension"][track] = result["state_dimension"]
                updates["fit_ok"][track] = 1
            diagnostics["tracks"].append(info)
        diagnostics["physical_3d"] = bool(physical and drift_error is None)
        diagnostics["successful_tracks"] = int(updates["fit_ok"].sum())
        missing = any(t.get("status") == "missing_calibration" for t in diagnostics["tracks"])
        diagnostics["status"] = ("ok" if k and np.all(updates["fit_ok"]) else
                                 "missing_calibration" if drift_error is not None else
                                 "partial_fit" if np.any(updates["fit_ok"]) else
                                 "missing_calibration" if missing else "no_valid_tracks")
        if missing:
            diagnostics["missing_inputs"] = sorted(set(key for track in diagnostics["tracks"]
                for key in track.get("missing_inputs", [])))
        if drift_error is not None:
            diagnostics.update(drift_error)
        return updates, diagnostics
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        diagnostics.update(status="invalid_input", message=str(error))
        return updates, diagnostics
