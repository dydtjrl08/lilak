"""Observation-only bridge tests; synthetic charge clouds are not TPC calibration."""
import copy
import unittest

import numpy as np

from reconstruction.algorithms.track_fit import fit_event
from reconstruction.preprocessing import PAD_U_MM, LAYER_V_MM


def configuration(route="direct"):
    return dict(route=route, drift=dict(calibrated_for_charge_mean=True,
                velocity_mm_per_ns=.01, t0_tb=200., reference_w_mm=120., sign=-1., tb_ns=40.),
                fitting=dict(method="linear_rts", process_noise_mode="none", minimum_layers=4))


def waveforms():
    q = np.zeros((2, 12, 64, 512), dtype=np.float32)
    for k in range(2):
        for layer in range(12):
            pad = 10 + layer if k == 0 else 40 + layer
            tb = 150 + 2 * layer if k == 0 else 250 - layer
            q[k, layer, pad, tb - 1:tb + 2] = [500., 1000., 500.]
    return dict(components=q, tb_ns=np.asarray(40.))


def point_record():
    points, covariance, charge, layer, fractions = [], [], [], [], []
    for track in range(2):
        for depth in range(12):
            u = 2. + .1 * LAYER_V_MM[depth] + 10. * track
            w = 50. - .04 * LAYER_V_MM[depth] + 20. * track
            for offset, amount in [(-.75, 1.), (.25, 3.)]:
                points.append([u + offset, LAYER_V_MM[depth], w + offset * 2.])
                covariance.append(np.diag([.02, 12., .1]))
                charge.append(amount); layer.append(depth)
                f = [0., 0., 0.]; f[track] = 1.; fractions.append(f)
    return dict(points_uvw_mm=np.array(points), point_covariance_mm2=np.array(covariance),
                point_charge=np.array(charge), point_layer=np.array(layer),
                point_fractions=np.array(fractions), physical_3d=np.asarray(1))


class TrackFitBridgeTests(unittest.TestCase):
    def test_direct_charge_centroid_timing_and_direction(self):
        updates, diagnostics = fit_event(waveforms(), configuration())
        self.assertEqual(diagnostics["status"], "ok", diagnostics)
        np.testing.assert_array_equal(updates["fit_ok"], [1, 1])
        self.assertEqual(updates["fit_states"].shape, (2, 12, 5))
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0, :, 0], PAD_U_MM[10:22])
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0, :, 2], 140. - .8 * np.arange(12.))
        np.testing.assert_allclose(updates["fit_layer_charge"], 2000.)
        expected = np.array([[1. / 12., 1., -.8 / 12.], [1. / 12., 1., .4 / 12.]])
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(updates["track_directions_uvw"], expected, atol=1e-8)
        self.assertTrue(np.all(np.isnan(updates["fit_energy_first_point_mev"])))
        self.assertFalse(diagnostics["component_assignment_uncertainty_included"])
        self.assertIn("NOT a calibrated", diagnostics["covariance_model"])

    def test_missing_drift_keeps_projected_diagnostic_but_never_3d(self):
        config = configuration(); config["drift"] = {}
        updates, diagnostics = fit_event(waveforms(), config)
        self.assertEqual(diagnostics["status"], "missing_calibration")
        self.assertFalse(np.any(updates["fit_ok"]))
        self.assertFalse(np.any(updates["fit_valid_layer"]))
        self.assertTrue(np.any(updates["fit_observed_layer_mask"]))
        self.assertTrue(np.isfinite(updates["fit_measurement_points_uv_mm"]).all())
        self.assertTrue(np.isfinite(updates["fit_measurement_covariance_uv_mm2"]).all())
        self.assertTrue(np.all(np.isnan(updates["fit_states"])))
        self.assertTrue(np.all(np.isnan(updates["track_directions_uvw"])))
        np.testing.assert_allclose(updates["fit_theta_pad_2d_ols_deg"],
                                   np.degrees(np.arctan(1. / 12.)), atol=1e-8)

    def test_missing_layers_stay_nan_and_reference_is_first_observation(self):
        record = waveforms(); record["components"][:, :3] = 0.
        updates, diagnostics = fit_event(record, configuration())
        self.assertEqual(diagnostics["status"], "ok")
        np.testing.assert_array_equal(updates["fit_valid_layer"][:, :3], 0)
        self.assertTrue(np.all(np.isnan(updates["fit_states"][:, :3])))
        self.assertTrue(np.all(np.isnan(updates["fit_points_uvw_mm"][:, :3])))
        np.testing.assert_allclose(updates["fit_reference_v_mm"], 36.)
        np.testing.assert_allclose(updates["track_origins_uvw_mm"][:, 1], 36.)

    def test_sparse_or_empty_components_do_not_fabricate_tracks(self):
        record = waveforms(); record["components"][:, 3:] = 0.
        updates, diagnostics = fit_event(record, configuration())
        self.assertEqual(diagnostics["status"], "no_valid_tracks")
        self.assertTrue(all(t["status"] == "insufficient_layers" for t in diagnostics["tracks"]))
        self.assertFalse(np.any(updates["fit_ok"]))
        record["components"][:] = 0.
        empty, diag = fit_event(record, configuration())
        self.assertFalse(np.any(empty["fit_ok"]))
        self.assertTrue(np.all(np.isnan(empty["fit_states"])))
        self.assertEqual(set(empty), set(updates))

    def test_cloud_width_is_not_divided_by_adc_amplitude(self):
        record = waveforms()
        first, _ = fit_event(record, configuration())
        record["components"] *= 100.
        second, _ = fit_event(record, configuration())
        np.testing.assert_allclose(first["fit_measurement_covariance_mm2"],
                                   second["fit_measurement_covariance_mm2"], atol=1e-12)
        np.testing.assert_allclose(second["fit_layer_charge"], first["fit_layer_charge"] * 100.)

    def test_pad_time_delay_and_tb_duration_validation(self):
        cfg = configuration(); cfg["drift"]["pad_delay_tb"] = np.full((12, 64), 5.)
        a, da = fit_event(waveforms(), cfg)
        b, db = fit_event(waveforms(), configuration())
        self.assertEqual(da["status"], "ok")
        np.testing.assert_allclose(a["fit_measurement_points_uvw_mm"][:, :, 2] -
                                   b["fit_measurement_points_uvw_mm"][:, :, 2], 2.)
        cfg["drift"]["tb_ns"] = 50.
        bad, diag = fit_event(waveforms(), cfg)
        self.assertEqual(diag["status"], "invalid_calibration")
        self.assertFalse(np.any(bad["fit_ok"]))

    def test_spacepoint_route_charge_weighting_and_cloud_covariance(self):
        record = point_record()
        record["components"] = np.zeros((1, 1))  # Must not be read on this route.
        updates, diagnostics = fit_event(record, configuration("spacepoint"))
        self.assertEqual(diagnostics["status"], "ok", diagnostics)
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0, :, 0], 2. + .1 * LAYER_V_MM)
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0, :, 2], 50. - .04 * LAYER_V_MM)
        np.testing.assert_allclose(updates["fit_layer_charge"], 4.)
        # Weighted offsets -.75/.25 have variance .1875, retained as spread proxy.
        np.testing.assert_allclose(updates["fit_measurement_covariance_mm2"][:, :, 0, 0], .02 + .1875)
        np.testing.assert_allclose(updates["fit_measurement_covariance_mm2"][:, :, 0, 2], .375)

    def test_spacepoint_physical_provenance_and_soft_fractions(self):
        record = point_record(); record["physical_3d"] = np.asarray(0)
        updates, diag = fit_event(record, configuration("spacepoint"))
        self.assertEqual(diag["status"], "missing_calibration")
        self.assertFalse(np.any(updates["fit_ok"]))
        record["physical_3d"] = np.asarray(1)
        record["point_fractions"][:] = [.25, .5, .25]
        updates, diag = fit_event(record, configuration("spacepoint"))
        self.assertEqual(diag["status"], "ok", diag)
        np.testing.assert_allclose(updates["fit_layer_charge"][1], 2. * updates["fit_layer_charge"][0])
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0],
                                   updates["fit_measurement_points_uvw_mm"][1])
        self.assertTrue(diag["conditional_fit_does_not_validate_track_separation"])

    def test_energy_aware_missing_prior_respected_and_explicit_priors_pass(self):
        cfg = configuration()
        cfg["fitting"].update(method="ekf_rts", energy_reference="first_point", energy_sigma_mev=.1,
                              straggling_variance_mev2_per_mm=0.,
                              initial_covariance=np.diag([100., 100., .01, .01, .01]).tolist(),
                              stopping_table=dict(energy_mev=[.001, 100.], dedx_mev_per_mm=[.001, .001],
                                  units=dict(energy="MeV", dedx="MeV/mm"), material="synthetic"))
        updates, diag = fit_event(waveforms(), cfg)
        self.assertFalse(np.any(updates["fit_ok"]))
        self.assertTrue(all(t["status"] == "missing_calibration" for t in diag["tracks"]))
        cfg["fitting"]["energy_first_point_mev"] = [20., 25.]
        updates, diag = fit_event(waveforms(), cfg)
        self.assertEqual(diag["status"], "ok", diag)
        np.testing.assert_allclose(updates["fit_energy_first_point_mev"], [20., 25.], atol=1e-8)
        np.testing.assert_array_equal(updates["fit_state_dimension"], [5, 5])

    def test_track_failure_is_separate_from_other_successful_tracks(self):
        cfg = configuration()
        cfg["fitting"]["per_track"] = [{}, dict(method="ukf_rts")]
        updates, diagnostics = fit_event(waveforms(), cfg)
        self.assertEqual(diagnostics["status"], "partial_fit")
        np.testing.assert_array_equal(updates["fit_ok"], [1, 0])
        self.assertEqual(diagnostics["tracks"][1]["status"], "missing_calibration")
        self.assertTrue(np.all(np.isnan(updates["fit_states"][1])))

    def test_spacepoint_nonzero_pad_delay_uses_embedded_observed_times(self):
        record = point_record(); cfg = configuration("spacepoint")
        original_w = record["points_uvw_mm"][:, 2]
        record["point_tb"] = 200. + (120. - original_w) / .4
        record["point_pad"] = np.zeros(len(original_w), dtype=np.int64)
        cfg["drift"]["pad_delay_tb"] = np.full(64, 3.)
        updates, diagnostics = fit_event(record, cfg)
        self.assertEqual(diagnostics["status"], "ok", diagnostics)
        np.testing.assert_allclose(updates["fit_measurement_points_uvw_mm"][0, :, 2],
                                   50. - .04 * LAYER_V_MM + 1.2)

    def test_input_arrays_unmodified_and_truth_ignored(self):
        record = waveforms(); config = configuration()
        before = record["components"].copy(); before_config = copy.deepcopy(config)
        expected, _ = fit_event(record, config)
        record["truth_theta"] = [9999., -9999.]
        record["si_alpha_mc_edep"] = [1e10, 1e10]
        actual, _ = fit_event(record, config)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)
        np.testing.assert_array_equal(record["components"], before)
        self.assertEqual(config, before_config)

    def test_invalid_observations_fail_with_constant_numeric_schema(self):
        reference, _ = fit_event(waveforms(), configuration())
        record = waveforms(); record["components"][0, 0, 0, 0] = -1.
        failed, diag = fit_event(record, configuration())
        self.assertEqual(diag["status"], "invalid_input")
        self.assertEqual(set(failed), set(reference))
        self.assertFalse(np.any(failed["fit_ok"]))
        record = point_record(); record["point_fractions"][0] = [.5, .5, .5]
        failed, diag = fit_event(record, configuration("spacepoint"))
        self.assertEqual(diag["status"], "invalid_input")
        self.assertFalse(np.any(failed["fit_ok"]))


if __name__ == "__main__":
    unittest.main()
