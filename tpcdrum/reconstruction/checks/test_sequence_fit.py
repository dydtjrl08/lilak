"""Legacy sequence adapter checks: synthetic geometry, never CNN accuracy claims."""
import copy
from pathlib import Path
import unittest

import numpy as np

from reconstruction.algorithms.sequence_fit import fit_event, _numpy_paths
from reconstruction.checks.test_track_fit import configuration, waveforms


class FixedObservedPredictor:
    metadata = dict(method="deterministic_SOFTWARE_TEST_STUB", trained=False)

    def __init__(self, angles=None, initial=None):
        self.angles = angles
        self.initial = initial

    def predict(self, components):
        k = len(components)
        peaks = np.max(components, axis=-1)
        mask = (peaks > 40.).any(axis=-1)
        return dict(theta_initial_deg=np.array([22., -24.])[:k] if self.initial is None else self.initial,
                    theta_segment_deg=np.full((k, 11), np.degrees(np.arctan(1. / 12.)))
                    if self.angles is None else np.asarray(self.angles),
                    layer_mask=mask, valid_segment=mask[:, :-1] & mask[:, 1:], peak_native=peaks)


def sequence_config(test=True):
    cfg = configuration()
    cfg["fitting"].update(method="legacy_sequence_cnn", legacy_sequence=dict(enabled=True))
    if test:
        cfg["fitting"]["legacy_sequence"]["allow_test_predictor"] = True
    return cfg


class SequenceFitTests(unittest.TestCase):
    def test_stub_geometry_uses_local_chord_not_initial_truth_head(self):
        output, diagnostics = fit_event(waveforms(), sequence_config(), FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "ok", diagnostics)
        self.assertFalse(diagnostics["trained_checkpoint_loaded"])
        self.assertIn("NOT_trained", diagnostics["model_source"])
        self.assertFalse(diagnostics["component_domain_validated"])
        np.testing.assert_array_equal(output["fit_ok"], [1, 1])
        expected = np.array([[1. / 12., 1., -.8 / 12.], [1. / 12., 1., .4 / 12.]])
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(output["track_directions_uvw"], expected, atol=1e-10)
        np.testing.assert_allclose(output["sequence_theta_initial_head_deg"], [22., -24.])
        np.testing.assert_allclose(output["sequence_theta_path_start_deg"],
                                   np.degrees(np.arctan(1. / 12.)))
        np.testing.assert_allclose(output["fit_points_uvw_mm"][0, :, 2], 140. - .8 * np.arange(12))

    def test_no_cnn_covariance_energy_or_kalman_pulls_are_fabricated(self):
        output, diagnostics = fit_event(waveforms(), sequence_config(), FixedObservedPredictor())
        self.assertTrue(np.all(np.isnan(output["fit_covariances"])))
        self.assertTrue(np.all(np.isnan(output["track_direction_covariances"])))
        self.assertTrue(np.all(np.isnan(output["fit_innovation_pulls"])))
        self.assertTrue(np.all(np.isnan(output["fit_states"][:, :, 4])))
        self.assertTrue(np.all(np.isnan(output["fit_energy_first_point_mev"])))
        self.assertFalse(diagnostics["cnn_covariance_available"])
        self.assertFalse(diagnostics["physics_accuracy_evaluated"])

    def test_full_sequence_path_integrates_varying_chords_and_peak_anchor(self):
        angles = np.tile(np.linspace(2., 10., 11), (2, 1))
        output, diagnostics = fit_event(waveforms(), sequence_config(), FixedObservedPredictor(angles))
        self.assertEqual(diagnostics["status"], "ok")
        path = output["sequence_projected_path_uv_mm"][0]
        np.testing.assert_allclose(np.diff(path[:, 0]), 12. * np.tan(np.deg2rad(angles[0])), atol=1e-10)
        centroid = np.arange(12.) - 26.5
        self.assertAlmostEqual(np.mean(centroid - path[:, 0]), 0., places=12)
        np.testing.assert_allclose(output["fit_states"][0, :11, 2], np.tan(np.deg2rad(angles[0])))
        self.assertAlmostEqual(output["fit_states"][0, -1, 2], np.tan(np.deg2rad(10.)))

    def test_longest_contiguous_run_and_first_observed_reference(self):
        record = waveforms(); record["components"][0, 5] = 0.
        output, diagnostics = fit_event(record, sequence_config(), FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "ok")
        np.testing.assert_array_equal(np.flatnonzero(output["fit_valid_layer"][0]), np.arange(6, 12))
        self.assertEqual(output["fit_reference_v_mm"][0], 72.)
        self.assertTrue(np.all(np.isnan(output["fit_points_uvw_mm"][0, :6])))
        record["components"][0, ::3] = 0.
        output, diagnostics = fit_event(record, sequence_config(), FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "partial_fit")
        np.testing.assert_array_equal(output["fit_ok"], [0, 1])

    def test_drift_missing_preserves_only_separate_projected_diagnostics(self):
        cfg = sequence_config(); cfg["drift"] = {}
        output, diagnostics = fit_event(waveforms(), cfg, FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "missing_calibration")
        self.assertFalse(np.any(output["fit_ok"]))
        self.assertTrue(np.all(np.isnan(output["fit_points_uvw_mm"])))
        self.assertTrue(np.all(np.isfinite(output["sequence_projected_path_uv_mm"])))
        self.assertTrue(np.all(np.isnan(output["track_directions_uvw"])))

    def test_trained_model_requires_explicit_opt_in_and_absolute_checkpoint(self):
        record = waveforms()
        disabled = sequence_config(False); disabled["fitting"]["legacy_sequence"]["enabled"] = False
        output, diagnostics = fit_event(record, disabled)
        self.assertEqual(diagnostics["status"], "model_not_enabled")
        self.assertFalse(np.any(output["fit_ok"]))
        missing = sequence_config(False)
        output, diagnostics = fit_event(record, missing)
        self.assertEqual(diagnostics["status"], "missing_model")
        missing["fitting"]["legacy_sequence"]["checkpoint"] = "best.pt"
        output, diagnostics = fit_event(record, missing)
        self.assertEqual(diagnostics["status"], "missing_model")
        output, diagnostics = fit_event(record, sequence_config(False), FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "test_predictor_not_enabled")

    def test_invalid_angles_masks_and_waveforms_do_not_silently_fit(self):
        angles = np.full((2, 11), 2.); angles[0, 0] = 90.
        output, diagnostics = fit_event(waveforms(), sequence_config(), FixedObservedPredictor(angles))
        self.assertEqual(diagnostics["status"], "partial_fit")
        np.testing.assert_array_equal(output["fit_ok"], [0, 1])
        self.assertEqual(diagnostics["tracks"][0]["status"], "invalid_forward_chord_angle")
        record = waveforms(); record["components"][0, 0, 0, 0] = -1.
        output, diagnostics = fit_event(record, sequence_config(), FixedObservedPredictor())
        self.assertEqual(diagnostics["status"], "invalid_prediction")
        self.assertFalse(np.any(output["fit_ok"]))

    def test_truth_fields_are_ignored_and_inputs_are_unmodified(self):
        record = waveforms(); config = sequence_config()
        before = record["components"].copy(); before_config = copy.deepcopy(config)
        first, _ = fit_event(record, config, FixedObservedPredictor())
        record["truth_initial_theta_deg"] = [999., -999.]
        record["mc_contribution_tags"] = np.ones(30)
        second, _ = fit_event(record, config, FixedObservedPredictor())
        for key in first:
            np.testing.assert_allclose(first[key], second[key], equal_nan=True)
        np.testing.assert_array_equal(record["components"], before)
        self.assertEqual(config, before_config)

    def test_actual_old_source_checkpoint_loading_and_path_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch unavailable; no untrained replacement is used")
        checkpoint = Path(__file__).resolve().parents[2] / "CnnTask/TrackReconstruction/root_reco_20261025/models/sequence_seed20260930.pt"
        if not checkpoint.is_file():
            self.skipTest("User's historical checkpoint is not in this source checkout")
        cfg = sequence_config(False)
        cfg["fitting"]["legacy_sequence"].update(checkpoint=str(checkpoint), device="cpu",
                    expected_data_fingerprint="92e9fec1047ad6b483bf9995d91eb70760c7521a1bd8c81fc4ccc1dbf5299db2")
        random_before = torch.get_rng_state().clone()
        output, diagnostics = fit_event(waveforms(), cfg)
        self.assertEqual(diagnostics["status"], "ok", diagnostics)
        self.assertTrue(diagnostics["trained_checkpoint_loaded"])
        self.assertEqual(diagnostics["model"]["checkpoint_sha256"],
                         "97fe05ce8df323a1ec4f62704e5e7f2f5776a94140748e4a1df1a9fe267947d5")
        self.assertTrue(diagnostics["model"]["composite_fingerprint_verified"])
        self.assertFalse(diagnostics["model"]["component_domain_validated"])
        self.assertTrue(torch.equal(random_before, torch.get_rng_state()))
        self.assertTrue(np.all(np.isnan(output["fit_covariances"])))
        from reconstruction.legacy_sequence.sequence_adapter import build_paths
        predictions = FixedObservedPredictor().predict(waveforms()["components"])
        actual, equivalent = build_paths(predictions), _numpy_paths(predictions)
        for a, b in zip(actual, equivalent):
            np.testing.assert_allclose(a["points_xy_mm"], b["points_xy_mm"])
            np.testing.assert_array_equal(a["layer_ids"], b["layer_ids"])


if __name__ == "__main__":
    unittest.main()
