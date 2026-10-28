"""Synthetic mathematics/software checks, never a detector-resolution claim.

Run: PYTHONPATH=tpcdrum:tmp/reconstruction_deps python -m unittest
     reconstruction.checks.test_smoother -v
"""
import copy
import unittest

import numpy as np

from reconstruction.algorithms.smoother import fit_track, _Transport


def energy_config(method="ekf_rts", **kwargs):
    result = dict(method=method, coordinate_system="physical_uvw_mm",
                  process_noise_mode="none", energy_reference="first_point",
                  energy_sigma_mev=.05, straggling_variance_mev2_per_mm=0.,
                  initial_covariance=np.diag([1e3, 1e3, .01, .01, .05 ** 2]).tolist())
    result.update(kwargs)
    return result


def toy_stopping(dedx=.01):
    return dict(energy_mev=[.001, 100.], dedx_mev_per_mm=[dedx, dedx],
                material="synthetic_constant_stopping_NOT_TPC_calibration",
                units=dict(energy="MeV", dedx="MeV/mm", length="mm"),
                provenance=dict(source="synthetic unit test"))


class SmootherTests(unittest.TestCase):
    def setUp(self):
        self.v = np.arange(12.) * 12.
        self.points = np.column_stack((2. + .1 * self.v, self.v, 20. - .05 * self.v))
        self.cov = np.diag([.04, .09])

    def assertOK(self, result):
        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["physics_accuracy_evaluated"])
        self.assertFalse(result["physical_uncertainty_validated"])

    def test_exact_line_closure_and_explicit_4d_fallback(self):
        result = fit_track(self.points[::-1], self.cov,
                           dict(method="linear_rts", process_noise_mode="none"))
        self.assertOK(result)
        self.assertEqual(result["state_dimension"], 4)
        np.testing.assert_allclose(result["fitted_points_uvw_mm"], self.points, atol=1e-10)
        np.testing.assert_allclose(result["states"][:, 2:4], np.tile([.1, -.05], (12, 1)), atol=1e-11)
        self.assertTrue(np.all(np.isnan(result["states"][:, 4])))
        self.assertTrue(np.all(np.isnan(result["covariances"][:, 4])))
        expected = np.array([.1, 1., -.05]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result["initial_downstream_direction"], expected, atol=1e-11)
        self.assertEqual(result["initial_reference_v_mm"], 0.)

    def test_zero_process_rts_matches_independent_weighted_least_squares(self):
        rng = np.random.default_rng(20261027)
        points = self.points.copy()
        points[:, (0, 2)] += rng.multivariate_normal([0., 0.], self.cov, len(points))
        result = fit_track(points, self.cov, dict(method="linear_rts", process_noise_mode="none"))
        self.assertOK(result)
        design = np.c_[np.ones(len(points)), self.v]
        ols = np.linalg.lstsq(design, points[:, (0, 2)], rcond=None)[0]
        np.testing.assert_allclose(result["states"][0, :4],
                                   np.r_[ols[0], ols[1]], rtol=1e-5, atol=2e-6)
        # Independent global normal matrix verifies cross-coordinate covariance.
        a = np.zeros((2 * len(points), 4))
        a[::2, 0] = 1.; a[1::2, 1] = 1.
        a[::2, 2] = self.v; a[1::2, 3] = self.v
        block_weight = np.kron(np.eye(len(points)), np.linalg.inv(self.cov))
        expected_cov = np.linalg.inv(a.T @ block_weight @ a)
        np.testing.assert_allclose(result["covariances"][0, :4, :4], expected_cov,
                                   rtol=2e-5, atol=2e-9)
        expected_chi2 = np.sum((points[:, (0, 2)] - design @ ols) ** 2 /
                               np.diag(self.cov))
        self.assertAlmostEqual(result["measurement_residual_chi2"], expected_chi2, places=6)

    def test_energy_transport_known_constant_stopping(self):
        expected = 20. - .01 * self.v * np.sqrt(1. + .1 ** 2 + .05 ** 2)
        result = fit_track(self.points, self.cov, energy_config(), 20., toy_stopping())
        self.assertOK(result)
        self.assertEqual(result["state_dimension"], 5)
        np.testing.assert_allclose(result["states"][:, 4], expected, rtol=0., atol=1e-8)
        self.assertTrue(np.all(np.isfinite(result["covariances"])))
        self.assertTrue(np.all(np.diff(result["states"][:, 4]) < 0.))
        for covariance in result["covariances"]:
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-9)

    def test_unscented_transform_improves_nonlinear_moment_prediction(self):
        # Nonlinearity is path length sqrt(1+tu^2+tw^2), not a fake B-field bend.
        cfg = energy_config()
        ekf = _Transport(cfg, 20., toy_stopping(.1))
        ukf = _Transport(energy_config("ukf_rts"), 20., toy_stopping(.1))
        state = np.array([0., 0., .2, -.1, 20.])
        covariance = np.diag([.01, .01, .09, .09, .0025])
        emean, ecov, _, _ = ekf.predict(state, covariance, 12.)
        umean, ucov, _, _ = ukf.predict(state, covariance, 12.)
        rng = np.random.default_rng(930)
        samples = rng.multivariate_normal(state, covariance, 150000)
        energies = samples[:, 4] - 1.2 * np.sqrt(1. + samples[:, 2] ** 2 + samples[:, 3] ** 2)
        empirical = energies.mean()
        self.assertLess(abs(umean[4] - empirical), abs(emean[4] - empirical) * .3)
        self.assertGreater(ecov[4, 4], 0.)
        self.assertGreater(ucov[4, 4], 0.)
        fit = fit_track(self.points, self.cov, energy_config("ukf_rts"), 20., toy_stopping())
        self.assertOK(fit)
        self.assertLess(abs(fit["states"][-1, 4] -
                            (20. - .01 * self.v[-1] * np.sqrt(1.0125))), .01)

    def test_curved_synthetic_path_needs_process_noise(self):
        rng = np.random.default_rng(42)
        v = np.arange(30.) * 6.
        truth = np.c_[.0008 * v ** 2 + .02 * v, v, 10. - .0006 * v ** 2]
        points = truth.copy(); points[:, (0, 2)] += rng.normal(0., .12, (len(v), 2))
        straight = fit_track(points, np.eye(2) * .12 ** 2,
                             dict(method="linear_rts", process_noise_mode="none"))
        local = fit_track(points, np.eye(2) * .12 ** 2,
                          dict(method="linear_rts", process_noise_mode="tuned",
                               angular_variance_rad2_per_mm=1e-4))
        self.assertOK(straight); self.assertOK(local)
        e0 = np.sqrt(np.mean((straight["fitted_points_uvw_mm"][:, (0, 2)] - truth[:, (0, 2)]) ** 2))
        e1 = np.sqrt(np.mean((local["fitted_points_uvw_mm"][:, (0, 2)] - truth[:, (0, 2)]) ** 2))
        self.assertLess(e1, e0 * .15)
        self.assertGreater(np.ptp(local["theta_pad_deg"]), 10.)

    def test_deterministic_annealing_rejects_observation_outliers(self):
        rng = np.random.default_rng(501)
        v = np.arange(30.) * 6.
        truth = np.c_[2. + .1 * v, v, 10. - .08 * v]
        points = truth.copy(); points[:, (0, 2)] += rng.normal(0., .15, (len(v), 2))
        bad = [3, 14, 24]; points[bad, 0] += [8., -10., 9.]
        points[bad, 2] += [-6., 12., 7.]
        plain = fit_track(points, np.eye(2) * .15 ** 2,
                          dict(method="linear_rts", process_noise_mode="none"))
        robust = fit_track(points, np.eye(2) * .15 ** 2,
                           dict(method="linear_rts", process_noise_mode="none",
                                robust=dict(kind="daf")))
        self.assertOK(plain); self.assertOK(robust)
        self.assertTrue(np.all(robust["daf_weights"][bad] < .01))
        keep = np.setdiff1d(np.arange(len(v)), bad)
        self.assertGreater(np.median(robust["daf_weights"][keep]), .95)
        e0 = np.sqrt(np.mean((plain["fitted_points_uvw_mm"] - truth) ** 2))
        e1 = np.sqrt(np.mean((robust["fitted_points_uvw_mm"] - truth) ** 2))
        self.assertLess(e1, e0 * .25)
        self.assertGreater(len(robust["annealing_history"]), 4)
        self.assertEqual(robust["annealing_history"][-1]["temperature"], 1.)
        self.assertTrue(np.all(np.isfinite(robust["leave_one_out_pulls"])))

    def test_energy_aware_ekf_ukf_with_annealing_and_nonlinear_stopping(self):
        points = self.points.copy(); points[5, (0, 2)] += [5., -6.]
        for method in ("daf_ekf", "daf_ukf"):
            fit = fit_track(points, self.cov, energy_config(method), 20., toy_stopping())
            self.assertOK(fit)
            self.assertEqual(fit["robust_method"], "daf")
            self.assertLess(fit["daf_weights"][5], .01)
            self.assertLess(np.max(np.abs(fit["fitted_points_uvw_mm"] - self.points)), .01)
            self.assertTrue(np.all(np.isfinite(fit["states"][:, 4])))
        # A linear S(T)=a*T has a known exponential solution, independent of RK4.
        table = toy_stopping(); table["dedx_mev_per_mm"] = [.000001, .1]
        fit = fit_track(self.points, self.cov, energy_config(), 20., table)
        self.assertOK(fit)
        np.testing.assert_allclose(fit["states"][:, 4],
                                   20. * np.exp(-.001 * self.v * np.sqrt(1.0125)), atol=1e-8)

    def test_tuned_scattering_synthetic_covariance_closure(self):
        # Known, synthetic model only. This is deliberately not TPC calibration.
        rng = np.random.default_rng(1027)
        config = dict(method="linear_rts", process_noise_mode="tuned",
                      angular_variance_rad2_per_mm=1.5e-5)
        transport = _Transport(config, None, None)
        f = np.eye(4); f[0, 2] = f[1, 3] = 12.
        standardized, covered, innovation = [], [], []
        mse_forward, mse_smooth = [], []
        for _ in range(80):
            true = np.zeros((12, 4)); true[0] = [0., 10., .08, -.03]
            for i in range(1, 12):
                q = transport.noise(true[i - 1], 12.)
                true[i] = f @ true[i - 1] + rng.multivariate_normal(np.zeros(4), q)
            points = np.c_[true[:, 0], self.v, true[:, 1]]
            points[:, (0, 2)] += rng.normal(0., .2, (12, 2))
            fit = fit_track(points, np.eye(2) * .04, config)
            self.assertOK(fit)
            error = fit["states"][0, 2:4] - true[0, 2:4]
            sigma = np.sqrt(np.diag(fit["covariances"][0])[2:4])
            standardized.extend(error / sigma); covered.extend(np.abs(error) < sigma)
            innovation.extend(fit["innovation_pulls"][2:].ravel())
            mse_forward.append(np.mean((fit["filtered_states"][:, :2] - true[:, :2]) ** 2))
            mse_smooth.append(np.mean((fit["states"][:, :2] - true[:, :2]) ** 2))
            traces_f = np.trace(fit["filtered_covariances"][:, :4, :4], axis1=1, axis2=2)
            traces_s = np.trace(fit["covariances"][:, :4, :4], axis1=1, axis2=2)
            self.assertTrue(np.all(traces_s <= traces_f + 1e-6))
        self.assertLess(abs(np.mean(standardized)), .3)
        self.assertTrue(.7 < np.std(standardized) < 1.35, np.std(standardized))
        self.assertTrue(.5 < np.mean(covered) < .85, np.mean(covered))
        self.assertTrue(.8 < np.std(innovation) < 1.2, np.std(innovation))
        self.assertLess(np.mean(mse_smooth), np.mean(mse_forward))

    def test_full_uvw_measurement_covariance_includes_v_uncertainty(self):
        small = fit_track(self.points, np.diag([.04, 0., .04]),
                          dict(method="linear_rts", process_noise_mode="none"))
        large = fit_track(self.points, np.diag([.04, 100., .04]),
                          dict(method="linear_rts", process_noise_mode="none"))
        self.assertOK(small); self.assertOK(large)
        self.assertGreater(large["covariances"][0, 2, 2], small["covariances"][0, 2, 2] * 10.)
        self.assertGreater(large["effective_measurement_covariances"][0, 0, 0], 1.)

    def test_energy_scattering_table_and_highland_domain_are_explicit(self):
        cfg = energy_config(process_noise_mode="table", scattering_table=dict(
            energy_mev=[.001, 100.], projected_angular_variance_rad2_per_mm=[1e-4, 1e-7],
            units=dict(energy="MeV", angular_variance="rad2/mm")))
        result = fit_track(self.points, self.cov, cfg, 20., toy_stopping())
        self.assertOK(result)
        self.assertGreater(result["process_covariances"][-1, 2, 2],
                           result["process_covariances"][1, 2, 2])
        cfg = energy_config(process_noise_mode="highland", material=dict(
            radiation_length_mm=1e6, mass_mev=3727.379, charge_e=2.))
        result = fit_track(self.points, self.cov, cfg, 20., toy_stopping())
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "outside_scattering_model_domain")
        cfg["allow_highland_outside_valid_range"] = True
        result = fit_track(self.points, self.cov, cfg, 20., toy_stopping())
        self.assertOK(result)
        self.assertIn("highland_outside_documented_thickness_range", result["warnings"])

    def test_missing_calibration_never_fabricates_energy_or_3d(self):
        missing_cov = fit_track(self.points, None, dict(method="linear_rts", process_noise_mode="none"))
        self.assertEqual(missing_cov["status"], "missing_calibration")
        for cfg, energy, table in [(energy_config(), None, toy_stopping()),
                                   (energy_config(), 20., None),
                                   (dict(method="linear_rts"), None, None)]:
            result = fit_track(self.points, self.cov, cfg, energy, table)
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "missing_calibration")
            self.assertTrue(result["missing_inputs"])
        cfg = energy_config(); cfg.pop("initial_covariance")
        result = fit_track(self.points, self.cov, cfg, 20., toy_stopping())
        self.assertEqual(result["status"], "missing_calibration")
        cfg = dict(method="linear_rts", process_noise_mode="none", coordinate_system="scaled_u_v_tb")
        result = fit_track(self.points, self.cov, cfg)
        self.assertEqual(result["status"], "missing_calibration")

    def test_input_validation_and_strict_stopping_domain(self):
        duplicate = self.points.copy(); duplicate[1, 1] = duplicate[0, 1]
        badcov = np.array([[1., 2.], [2., 1.]])
        for points, covariance in [(duplicate, self.cov), (self.points, badcov),
                                   (self.points[:2], self.cov)]:
            self.assertFalse(fit_track(points, covariance,
                                      dict(method="linear_rts", process_noise_mode="none"))["success"])
        table = toy_stopping(); table["units"]["dedx"] = "MeV/cm"
        result = fit_track(self.points, self.cov, energy_config(), 20., table)
        self.assertEqual(result["status"], "missing_calibration")
        table = toy_stopping(.5); table["energy_mev"] = [.001, 1.]
        result = fit_track(self.points, self.cov, energy_config(), .5, table)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "transport_domain_error")
        # Normalized PSD checking must not let a huge variance conceal a
        # negative mode in a small-variance component.
        multiscale_bad = np.array([[1e6, 0., 0.], [0., 1., .02], [0., .02, 1e-8]])
        result = fit_track(self.points, multiscale_bad,
                           dict(method="linear_rts", process_noise_mode="none"))
        self.assertFalse(result["success"])

    def test_does_not_mutate_observations_config_or_tables(self):
        points, cov = self.points.copy(), self.cov.copy()
        config, table = energy_config(), toy_stopping()
        before_config, before_table = copy.deepcopy(config), copy.deepcopy(table)
        self.assertOK(fit_track(points, cov, config, 20., table))
        np.testing.assert_array_equal(points, self.points)
        np.testing.assert_array_equal(cov, self.cov)
        self.assertEqual(config, before_config); self.assertEqual(table, before_table)


if __name__ == "__main__":
    unittest.main()
