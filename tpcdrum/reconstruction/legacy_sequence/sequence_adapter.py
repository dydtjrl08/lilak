"""Observed-signal inference for the 2026-10-22 single-alpha sequence model.

The CNN predicts projected x-y angles, not 3D positions. ``build_paths`` adds a
measured translation to integrated chord slopes. A separate timing calibration
is required before using these paths for a 3D vertex. No MC input is accepted.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch

from .sequence_models import build_model
from .checkpoint_io import load_checkpoint


LAYER_Y_MM = np.arange(12, dtype=np.float64) * 12.0
PAD_X_MM = np.r_[np.arange(27) - 36.5, np.arange(10) * 2.0 - 9.0,
                 np.arange(27) + 10.5]
GRID_ROWS = np.r_[np.arange(27), np.repeat(np.arange(27, 37), 2), np.arange(37, 64)]
GRID_WEIGHT = np.r_[np.ones(27), np.full(20, 0.5), np.ones(27)].astype(np.float32)
ADC_SCALE = 4095.0
ACTIVITY_THRESHOLD_ADC = 40.0


def _load_checkpoint(path, trusted_checkpoint):
    return load_checkpoint(path, trusted_checkpoint)


def _check_preprocessing(preprocessing):
    """Validate supplied input_audit metadata against this fixed adapter."""
    declared = dict(preprocessing or {})
    info = declared.get("sequence_measurements", declared)
    required = {
        "adc_scale": ADC_SCALE, "activity_threshold_adc": ACTIVITY_THRESHOLD_ADC,
        "pedestal": "median_raw_TB_0_29", "input": "peak_ADC_only",
        "images": "physical_1mm_grid_74_bins_charge_conserving_wide_pad_remap",
    }
    for name, expected in required.items():
        if name in info and info[name] != expected:
            raise ValueError("Unsupported sequence preprocessing %s=%r; expected %r" %
                             (name, info[name], expected))
    return required


def preprocess_components(component_waveforms):
    """Full-TB, nonnegative, pedestal-corrected separated ADC -> model input.

    max_t must be taken BEFORE the native64-to-image74 remap. Time-binned sums,
    normalized fractions and MC contribution tags are not waveform substitutes.
    Pedestal subtraction is performed once upstream on the observed waveform.
    """
    waves = np.asarray(component_waveforms)
    if waves.ndim != 4 or waves.shape[1:] != (12, 64, 512):
        raise ValueError("component_waveforms must have shape [K,12,64,512]")
    if not np.issubdtype(waves.dtype, np.number) or not np.isfinite(waves).all():
        raise ValueError("Component waveforms must be finite numeric ADC values")
    if np.any(waves < 0):
        raise ValueError("Component waveforms must already be pedestal-corrected and nonnegative")
    peaks = np.max(waves, axis=-1).astype(np.float32)
    if not np.isfinite(peaks).all():
        raise ValueError("Component peak ADC overflows float32")
    images = peaks[:, :, GRID_ROWS][:, None]
    images = np.ascontiguousarray(images * GRID_WEIGHT[None, None, None, :] / ADC_SCALE,
                                 dtype=np.float32)
    mask = (peaks > ACTIVITY_THRESHOLD_ADC).any(axis=-1)
    return images, mask, peaks


class SequenceFitter:
    """Load one trained resnet_sequence checkpoint once for event-loop inference.

    The default is restricted torch loading. Set trusted_checkpoint=True only
    for your own original training checkpoint (contains optimizer/RNG objects).
    A safe inference-only dict with model tensors needs no opt-in.
    expected_data_fingerprint checks the composite training fingerprint, which
    is distinct from a measurement-only preprocessing['fingerprint'] audit.
    """

    def __init__(self, checkpoint_path, device="cpu", trusted_checkpoint=False,
                 preprocessing=None, expected_data_fingerprint=None):
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Trained sequential-theta checkpoint not found: " + str(path))
        checkpoint = _load_checkpoint(path, trusted_checkpoint)
        if not isinstance(checkpoint, dict):
            raise ValueError("Sequence checkpoint must be a dictionary")
        if checkpoint.get("method") != "resnet_sequence":
            raise ValueError("Expected a trained resnet_sequence checkpoint; scalar theta models are incompatible")
        if checkpoint.get("output_units") != "degree":
            raise ValueError("Sequence checkpoint must explicitly declare output_units='degree'")
        config = checkpoint.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("Sequence checkpoint is missing model_config")
        if config.get("n_layers") != 12 or config.get("n_pads") != 74:
            raise ValueError("Sequence checkpoint geometry must explicitly specify n_layers=12, n_pads=74")
        y = np.asarray(config.get("layer_y_mm", []), dtype=float)
        if y.shape != (12,) or not np.allclose(y, LAYER_Y_MM, rtol=0, atol=1e-8):
            raise ValueError("Sequence checkpoint layer_y_mm must be 0,12,...,132 mm")
        embedded_preprocessing = checkpoint.get("preprocessing")
        if embedded_preprocessing is not None:
            if not isinstance(embedded_preprocessing, dict):
                raise ValueError("Checkpoint preprocessing must be a dictionary")
            _check_preprocessing(embedded_preprocessing)
        contract = _check_preprocessing(preprocessing)
        audit_fingerprint = (preprocessing or {}).get("fingerprint")
        if (expected_data_fingerprint is not None
                and checkpoint.get("data_fingerprint") != expected_data_fingerprint):
            raise ValueError("Checkpoint and expected composite training data fingerprint differ")
        # Training data_fingerprint also hashes truth, targets, groups and
        # splits. It cannot verify a measurement-only input_audit fingerprint.
        # Original 20261022 checkpoints have no separate measurement identity.
        measurement_fingerprint = checkpoint.get("measurement_fingerprint")
        training_contract = checkpoint.get("contract")
        if isinstance(training_contract, dict):
            contract_measurement = training_contract.get("measurement_fingerprint")
            if contract_measurement is not None:
                if (measurement_fingerprint is not None
                        and measurement_fingerprint != contract_measurement):
                    raise ValueError("Checkpoint measurement fingerprint declarations conflict")
                measurement_fingerprint = contract_measurement
        measurement_verified = (audit_fingerprint is not None
                                and measurement_fingerprint is not None)
        if measurement_verified and audit_fingerprint != measurement_fingerprint:
            raise ValueError("Checkpoint and supplied measurement audit fingerprint differ")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu' explicitly")
        # Construction initializes weights before loading them; preserve the
        # caller's random stream (e.g. the subsequent Hough/RANSAC sampling).
        with torch.random.fork_rng(devices=[]):
            self.model = build_model("resnet_sequence", config)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        # The geometry buffer in the weights must agree with constructor data.
        normalized_y = (LAYER_Y_MM - LAYER_Y_MM.mean()) / 66.0
        actual = self.model.layer_y_normalized.detach().cpu().numpy()
        if actual.shape != (12,) or not np.allclose(actual, normalized_y, rtol=0, atol=1e-7):
            raise ValueError("Checkpoint geometry buffer conflicts with its model_config")
        self.model.to(self.device).eval()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.metadata = {
            "checkpoint_path": str(path), "checkpoint_sha256": digest.hexdigest(),
            "method": "resnet_sequence", "output_units": "degree",
            "model_config": config, "preprocessing": contract,
            "data_fingerprint": checkpoint.get("data_fingerprint"),
            "composite_fingerprint_verified": expected_data_fingerprint is not None,
            "measurement_audit_fingerprint": audit_fingerprint,
            "checkpoint_measurement_fingerprint": measurement_fingerprint,
            "measurement_audit_fingerprint_verified": measurement_verified,
            "input_audit_fingerprint_verified": measurement_verified,
            "preprocessing_source": ("checkpoint" if embedded_preprocessing is not None
                                     else "fixed_20261022_adapter_contract"),
            "contract_hash": checkpoint.get("contract_hash"),
            "device": str(self.device), "trained_input": "single_alpha_peak_ADC",
            "component_domain_validated": False,
            "projection": "theta=atan2(delta_pad_x,delta_pad_y)",
        }

    def predict(self, component_waveforms):
        images, mask, peaks = preprocess_components(component_waveforms)
        count = len(images)
        if count:
            with torch.no_grad():
                out = self.model(torch.from_numpy(images).to(self.device),
                                 layer_mask=torch.from_numpy(mask).to(self.device))
            initial = out["initial"].detach().float().cpu().numpy()
            segment = out["segment"].detach().float().cpu().numpy()
        else:
            initial, segment = np.empty(0, np.float32), np.empty((0, 11), np.float32)
        if initial.shape != (count,) or segment.shape != (count, 11):
            raise RuntimeError("Sequence model returned unexpected output shape")
        if not np.isfinite(initial).all() or not np.isfinite(segment).all():
            raise FloatingPointError("Sequence model produced non-finite angle predictions")
        return {"theta_initial_deg": initial, "theta_segment_deg": segment,
                "layer_mask": mask, "valid_segment": mask[:, :-1] & mask[:, 1:],
                "peak_native": peaks}


def build_paths(predictions, min_layers=4):
    """Integrate the longest contiguous observed run and anchor it to centroids.

    Uses all positive native-pad component peaks for each layer centroid and
    gives each included layer equal weight when fitting the translation. The
    first chord is the local upstream direction; the separate initial-angle
    head is exposed for diagnostics, not silently substituted. Returned paths
    contain only observed y support, no extrapolation and no z coordinate.
    """
    if isinstance(min_layers, bool) or int(min_layers) != min_layers or not 4 <= min_layers <= 12:
        raise ValueError("min_layers must be an integer from 4 to 12 (trained cohort)")
    peaks = np.asarray(predictions["peak_native"], dtype=np.float64)
    mask = np.asarray(predictions["layer_mask"], dtype=bool)
    angles = np.asarray(predictions["theta_segment_deg"], dtype=np.float64)
    initial = np.asarray(predictions["theta_initial_deg"], dtype=np.float64)
    valid = np.asarray(predictions["valid_segment"], dtype=bool)
    count = len(peaks)
    if (peaks.shape != (count, 12, 64) or mask.shape != (count, 12)
            or angles.shape != (count, 11) or initial.shape != (count,)
            or valid.shape != (count, 11)):
        raise ValueError("Prediction shapes do not match sequence inference output")
    if not np.isfinite(peaks).all() or np.any(peaks < 0):
        raise ValueError("Invalid component peaks")
    if not np.array_equal(mask, (peaks > ACTIVITY_THRESHOLD_ADC).any(-1)):
        raise ValueError("Layer mask must be the observed native-peak >40 ADC mask")
    if np.any(valid & ~(mask[:, :-1] & mask[:, 1:])):
        raise ValueError("A valid segment cannot bridge an unobserved layer")
    results = []
    for index in range(count):
        item = {"track_index": index, "status": "insufficient_contiguous_layers",
                "theta_initial_deg": float(initial[index]), "points_xy_mm": np.empty((0, 2)),
                "layer_ids": np.empty(0, dtype=np.int64), "theta_start_deg": float("nan"),
                "centroid_x_mm": np.empty(0), "anchor_offset_mm": float("nan"),
                "centroid_residual_rms_mm": float("nan")}
        runs, run = [], []
        for layer in range(12):
            if mask[index, layer]:
                if run and not valid[index, layer-1]:
                    runs.append(run); run = []
                run.append(layer)
            elif run:
                runs.append(run); run = []
        if run:
            runs.append(run)
        ids = np.asarray(max(runs, key=len) if runs else [], dtype=np.int64)
        if len(ids) < min_layers:
            results.append(item); continue
        theta = angles[index, ids[:-1]]
        # θ is a forward-y projected chord. Beyond +/-90 it cannot represent
        # that training target, and tan near its pole is not a usable path.
        if not np.isfinite(theta).all() or np.any(np.abs(theta) >= 89.0):
            item["status"] = "invalid_forward_chord_angle"
            results.append(item); continue
        weights = peaks[index, ids]
        centroid = (weights * PAD_X_MM[None]).sum(-1) / weights.sum(-1)
        y = LAYER_Y_MM[ids]
        relative_x = np.r_[0.0, np.cumsum(np.diff(y) * np.tan(np.deg2rad(theta)))]
        offset = float(np.mean(centroid - relative_x))
        x = relative_x + offset
        item.update(status="ok", points_xy_mm=np.column_stack((x, y)), layer_ids=ids,
                    theta_start_deg=float(theta[0]), centroid_x_mm=centroid,
                    anchor_offset_mm=offset,
                    centroid_residual_rms_mm=float(np.sqrt(np.mean((centroid-x)**2))))
        results.append(item)
    return results
