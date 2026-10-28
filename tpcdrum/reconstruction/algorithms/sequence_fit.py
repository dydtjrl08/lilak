"""Explicitly opted-in source adapter for the user's legacy sequence ResNet.

The network predicts projected u-v chords, not physical w and not covariance.
Physical w is assembled from the current observed components and explicit
charge-mean timing calibration. No random/untrained network is substituted.
Already-run output stages do not need this checkpoint to be reloaded later.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ..preprocessing import PAD_U_MM, LAYER_V_MM
from .track_fit import _empty, _drift, _waveform_measurements


def _outputs(k):
    result = _empty(k)
    result.update(sequence_theta_initial_head_deg=np.full(k, np.nan),
                  sequence_theta_segment_deg=np.full((k, 11), np.nan),
                  sequence_theta_path_start_deg=np.full(k, np.nan),
                  sequence_valid_segment=np.zeros((k, 11), dtype=np.uint8),
                  sequence_projected_path_uv_mm=np.full((k, 12, 2), np.nan),
                  sequence_centroid_residual_rms_mm=np.full(k, np.nan),
                  sequence_anchor_offset_mm=np.full(k, np.nan))
    return result


def _numpy_paths(predictions, min_layers=4):
    """NumPy-only equivalent of legacy build_paths for injected test predictors.

    Production uses the source package's original build_paths. This fallback
    lets deterministic bridge tests run without importing/installing torch.
    """
    peaks = np.asarray(predictions["peak_native"], dtype=float)
    mask = np.asarray(predictions["layer_mask"], dtype=bool)
    angles = np.asarray(predictions["theta_segment_deg"], dtype=float)
    initial = np.asarray(predictions["theta_initial_deg"], dtype=float)
    valid = np.asarray(predictions["valid_segment"], dtype=bool)
    results = []
    for track in range(len(peaks)):
        item = dict(track_index=track, status="insufficient_contiguous_layers",
                    theta_initial_deg=float(initial[track]), points_xy_mm=np.empty((0, 2)),
                    layer_ids=np.empty(0, dtype=np.int64), theta_start_deg=float("nan"),
                    centroid_x_mm=np.empty(0), anchor_offset_mm=float("nan"),
                    centroid_residual_rms_mm=float("nan"))
        runs, run = [], []
        for layer in range(12):
            if mask[track, layer]:
                if run and not valid[track, layer - 1]:
                    runs.append(run); run = []
                run.append(layer)
            elif run:
                runs.append(run); run = []
        if run:
            runs.append(run)
        ids = np.asarray(max(runs, key=len) if runs else [], dtype=np.int64)
        if len(ids) >= min_layers:
            theta = angles[track, ids[:-1]]
            if not np.isfinite(theta).all() or np.any(np.abs(theta) >= 89.):
                item["status"] = "invalid_forward_chord_angle"
            else:
                weights = peaks[track, ids]
                centroid = (weights * PAD_U_MM[None]).sum(-1) / weights.sum(-1)
                v = LAYER_V_MM[ids]
                relative = np.r_[0., np.cumsum(np.diff(v) * np.tan(np.deg2rad(theta)))]
                offset = float(np.mean(centroid - relative))
                path_u = relative + offset
                item.update(status="ok", points_xy_mm=np.c_[path_u, v], layer_ids=ids,
                            theta_start_deg=float(theta[0]), centroid_x_mm=centroid,
                            anchor_offset_mm=offset,
                            centroid_residual_rms_mm=float(np.sqrt(np.mean((centroid - path_u) ** 2))))
        results.append(item)
    return results


@lru_cache(maxsize=2)
def _load_source_model(path, bytes_count, mtime_ns, device, trusted, preprocessing_json, fingerprint):
    """Cache only an explicitly configured current-stage checkpoint, never outputs."""
    from ..legacy_sequence.sequence_adapter import SequenceFitter, build_paths
    fitter = SequenceFitter(path, device=device, trusted_checkpoint=trusted,
                            preprocessing=json.loads(preprocessing_json),
                            expected_data_fingerprint=fingerprint)
    return fitter, build_paths


def _prediction_contract(predictions, components):
    k = len(components)
    shapes = dict(theta_initial_deg=(k,), theta_segment_deg=(k, 11),
                  layer_mask=(k, 12), valid_segment=(k, 11), peak_native=(k, 12, 64))
    for key, shape in shapes.items():
        if key not in predictions or np.asarray(predictions[key]).shape != shape:
            raise ValueError("Invalid sequence prediction field/shape: " + key)
    peaks = np.max(components, axis=-1)
    if (not np.allclose(predictions["peak_native"], peaks, rtol=1e-6, atol=1e-6) or
            not np.array_equal(predictions["layer_mask"], np.any(peaks > 40., axis=-1))):
        raise ValueError("Sequence peaks/mask must be derived from the observed native ADC >40")
    mask = np.asarray(predictions["layer_mask"], dtype=bool)
    if np.any(np.asarray(predictions["valid_segment"], dtype=bool) & ~(mask[:, :-1] & mask[:, 1:])):
        raise ValueError("Sequence valid segment cannot bridge an unobserved layer")
    if not np.all(np.isfinite(predictions["theta_initial_deg"])):
        raise ValueError("Nonfinite initial-angle head output")


def fit_event(record, config, predictor=None):
    """Return the same fit_* geometry schema as track_fit.fit_event.

    Required explicit configuration:
      fitting.method = legacy_sequence_cnn (or sequence_cnn/resnet_sequence)
      fitting.legacy_sequence.enabled = true
      fitting.legacy_sequence.checkpoint = absolute source checkpoint path
    device/trusted_checkpoint/preprocessing/expected_data_fingerprint are passed
    to the original safe source adapter. A missing torch/checkpoint is a failure.

    ``predictor`` is dependency injection for deterministic SOFTWARE tests only;
    it requires legacy_sequence.allow_test_predictor=true and is labelled as
    NOT a trained checkpoint. Its only API is predict(observed components).
    CNN state/direction covariances, Kalman pulls and energy remain unavailable.
    """
    k = 0
    if isinstance(record, Mapping) and "components" in record and np.ndim(record["components"]) == 4:
        k = np.shape(record["components"])[0]
    output = _outputs(k)
    diagnostics = dict(status="invalid_input", tracks=[], method="legacy_sequence_cnn",
                       physical_3d=False, trained_checkpoint_loaded=False,
                       physics_accuracy_evaluated=False, physical_uncertainty_validated=False,
                       component_domain_validated=False, cnn_covariance_available=False,
                       covariance_note="CNN trajectory and direction covariance are NOT calibrated; NaN is retained.",
                       geometry_note="Projected CNN u-v chords plus observed calibrated charge-mean w; this is not a 3D-trained CNN.",
                       first_direction_definition="first observed downstream chord, not the production direction or separate initial-angle head")
    try:
        if not isinstance(record, Mapping) or not isinstance(config, Mapping):
            raise ValueError("record and config must be mappings")
        fitting = dict(config.get("fitting", {}))
        options = dict(fitting.get("legacy_sequence", {}))
        if (fitting.get("method") not in ("legacy_sequence_cnn", "sequence_cnn", "resnet_sequence") or
                options.get("enabled") is not True):
            diagnostics.update(status="model_not_enabled", message="Explicit legacy-sequence method and enabled=true are required")
            return output, diagnostics
        components = np.asarray(record["components"])
        if (components.shape != (k, 12, 64, 512) or not np.all(np.isfinite(components)) or
                np.any(components < 0.)):
            raise ValueError("Full finite nonnegative observed components (K,12,64,512) are required")
        minimum = options.get("minimum_contiguous_layers", fitting.get("minimum_layers", 4))
        if isinstance(minimum, bool) or int(minimum) != minimum or not 4 <= minimum <= 12:
            raise ValueError("Legacy trained-cohort minimum contiguous layers must be 4..12")
        minimum = int(minimum)
        if predictor is not None:
            if options.get("allow_test_predictor") is not True:
                diagnostics.update(status="test_predictor_not_enabled",
                                   message="Injected predictors require explicit allow_test_predictor; no untrained fallback")
                return output, diagnostics
            builder = _numpy_paths
            diagnostics["model_source"] = "injected_test_predictor_NOT_trained_checkpoint"
            diagnostics["model"] = copy.deepcopy(getattr(predictor, "metadata", {}))
        else:
            filename = options.get("checkpoint")
            if not filename:
                diagnostics.update(status="missing_model", missing_inputs=["absolute_legacy_sequence_checkpoint"])
                return output, diagnostics
            path = Path(filename).expanduser()
            if not path.is_absolute():
                diagnostics.update(status="missing_model", message="Use an explicit absolute checkpoint path")
                return output, diagnostics
            path = path.resolve()
            if not path.is_file():
                diagnostics.update(status="missing_model", message="Checkpoint file does not exist: " + str(path))
                return output, diagnostics
            stat = path.stat()
            try:
                predictor, builder = _load_source_model(str(path), stat.st_size, stat.st_mtime_ns,
                    str(options.get("device", "cpu")), bool(options.get("trusted_checkpoint", False)),
                    json.dumps(options.get("preprocessing", {}), sort_keys=True, allow_nan=False),
                    options.get("expected_data_fingerprint"))
            except ImportError as error:
                diagnostics.update(status="missing_dependency", message=str(error), missing_inputs=["torch"])
                return output, diagnostics
            except (ValueError, RuntimeError, KeyError) as error:
                diagnostics.update(status="invalid_model", message=str(error))
                return output, diagnostics
            diagnostics.update(trained_checkpoint_loaded=True,
                               model_source="explicit_source_checkpoint", model=copy.deepcopy(predictor.metadata))
        predictions = predictor.predict(components)
        _prediction_contract(predictions, components)
        paths = builder(predictions, min_layers=minimum)
        output["sequence_theta_initial_head_deg"] = np.asarray(predictions["theta_initial_deg"], dtype=float)
        output["sequence_theta_segment_deg"] = np.asarray(predictions["theta_segment_deg"], dtype=float)
        output["sequence_valid_segment"] = np.asarray(predictions["valid_segment"], dtype=np.uint8)
        calibration, drift_error = _drift(config.get("drift"), record)
        measurements, measured_cov, charge, measured_valid = _waveform_measurements(record, {}, calibration)
        output["fit_layer_charge"] = charge
        output["fit_measurement_points_uv_mm"] = measurements[:,:,:2].copy()
        output["fit_measurement_covariance_uv_mm2"] = measured_cov[:,:,:2,:2].copy()
        output["fit_observed_layer_mask"] = measured_valid.astype(np.uint8)
        for track, path in enumerate(paths):
            info = dict(track=track, status=path["status"], minimum_contiguous_layers=minimum,
                        component_domain_validated=False, physical_uncertainty_validated=False,
                        trained_checkpoint_loaded=diagnostics["trained_checkpoint_loaded"])
            if path["status"] != "ok":
                diagnostics["tracks"].append(info)
                continue
            ids = np.asarray(path["layer_ids"], dtype=int)
            uv = np.asarray(path["points_xy_mm"], dtype=float)
            output["sequence_projected_path_uv_mm"][track, ids] = uv
            output["sequence_theta_path_start_deg"][track] = path["theta_start_deg"]
            output["sequence_centroid_residual_rms_mm"][track] = path["centroid_residual_rms_mm"]
            output["sequence_anchor_offset_mm"][track] = path["anchor_offset_mm"]
            info.update(observed_layers=ids.tolist(), anchor_offset_mm=path["anchor_offset_mm"],
                        anchor_definition="equal-layer-weight translation against native peak-ADC centroids",
                        centroid_residual_rms_mm=path["centroid_residual_rms_mm"],
                        theta_initial_head_deg=float(predictions["theta_initial_deg"][track]),
                        theta_first_chord_deg=path["theta_start_deg"],
                        last_node_slope_definition="preceding chord; not a measured differential tangent")
            design = np.c_[np.ones(len(ids)), uv[:, 1]]
            output["fit_theta_pad_2d_ols_deg"][track] = np.degrees(np.arctan(
                np.linalg.lstsq(design, uv[:, 0], rcond=None)[0][1]))
            if drift_error is not None:
                info.update(drift_error)
                diagnostics["tracks"].append(info)
                continue
            points = np.c_[uv, measurements[track, ids, 2]]
            delta = np.diff(points, axis=0)
            slopes = delta[:, (0, 2)] / delta[:, 1, None]
            node_slopes = np.vstack((slopes, slopes[-1]))
            direction = delta[0] / np.linalg.norm(delta[0])
            output["fit_states"][track, ids, :4] = np.c_[points[:, 0], points[:, 2], node_slopes]
            output["fit_points_uvw_mm"][track, ids] = points
            output["fit_measurement_points_uvw_mm"][track, ids] = measurements[track, ids]
            output["fit_measurement_covariance_mm2"][track, ids] = measured_cov[track, ids]
            output["fit_valid_layer"][track, ids] = 1
            output["fit_reference_v_mm"][track] = points[0, 1]
            output["fit_state_dimension"][track] = 4
            output["track_origins_uvw_mm"][track] = points[0]
            output["track_directions_uvw"][track] = direction
            output["fit_ok"][track] = 1
            # No chi2/pulls, DAF memberships or CNN uncertainty are fabricated.
            info["state_covariance_status"] = "unavailable_for_legacy_CNN"
            diagnostics["tracks"].append(info)
        diagnostics.update(physical_3d=calibration is not None, successful_tracks=int(output["fit_ok"].sum()))
        diagnostics["status"] = ("missing_calibration" if drift_error is not None else
                                 "ok" if k and np.all(output["fit_ok"]) else
                                 "partial_fit" if np.any(output["fit_ok"]) else "no_valid_tracks")
        if drift_error is not None:
            diagnostics.update(drift_error)
        return output, diagnostics
    except OSError as error:
        diagnostics.update(status="model_io_error", message=str(error))
        return output, diagnostics
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError, FloatingPointError) as error:
        diagnostics.update(status="invalid_prediction", message=str(error))
        return output, diagnostics
