"""Synthetic common-origin mathematics checks, not detector/production-angle validation."""
import copy
import unittest

import numpy as np

from reconstruction.algorithms.joint_vertex import vertex_region, joint_fit


def config(dimension=3, k=3):
    return dict(dimension=dimension, physical_3d=dimension == 3,
                vertex_v_scan_mm=[-200., 140.], process_noise_mode='none', scan_points=65,
                track_source_groups=[101]*k, track_fragment_ids=list(range(k)))


def observations(dimension=3, noise=False):
    vertex = np.array([1., -20., 50.]) if dimension == 3 else np.array([1., -20.])
    slopes = np.array([[-.12, .08], [.15, -.1], [.04, .15]]) if dimension == 3 else np.array([[-.12], [.15]])
    axes = [0, 2] if dimension == 3 else [0]
    tracks, covariances = [], []
    rng = np.random.default_rng(10027)
    for i, slope in enumerate(slopes):
        v = np.arange(8.)*12.+i*12.
        points = np.ones((len(v), dimension)); points[:, 1] = v
        points[:, axes] = vertex[axes]+(v-vertex[1])[:, None]*slope
        covariance = np.diag([.04, 0., .09]) if dimension == 3 else np.diag([.04, 0.])
        if dimension == 3:covariance[0, 2] = covariance[2, 0] = .025
        if noise:
            points += rng.multivariate_normal(np.zeros(dimension), covariance, len(v))
        tracks.append(points); covariances.append(np.tile(covariance, (len(v), 1, 1)))
    return vertex, tracks, covariances


class JointVertexTests(unittest.TestCase):
    def test_region_local_full_state_covariance_and_first_hit_gaps(self):
        vertex, tracks, _ = observations()
        origins = np.stack([p[0] for p in tracks])
        directions = np.stack([p[1]-p[0] for p in tracks])
        covariance = np.tile(np.diag([.04, .09, 1e-5, 2e-5]), (3, 1, 1))
        covariance[:, 0, 2] = covariance[:, 2, 0] = -.0003
        region = vertex_region(origins, directions, covariance, origins[:, 1], config())
        self.assertEqual(region['status'], 'resolved', region)
        np.testing.assert_allclose(region['point'], vertex, atol=1e-5)
        self.assertFalse(any(region['open_v_sides']))
        self.assertTrue(np.isfinite(region['covariance']).all())
        self.assertFalse(region['physical_uncertainty_validated'])
        wrong = vertex_region(origins, directions, covariance, origins[:, 1]+1., config())
        self.assertEqual(wrong['status'], 'invalid_input')

    def test_parallel_and_nearly_parallel_produce_regions_not_fake_points(self):
        origin = np.array([[-2., 0.], [2., 24.]])
        for slopes in ([0., 0.], [.00001, -.00001]):
            directions = np.c_[slopes, np.ones(2)]
            covariance = np.tile(np.diag([.04, 1e-6]), (2, 1, 1))
            region = vertex_region(origin, directions, covariance, origin[:, 1], config(2, 2), dimension=2)
            self.assertEqual(region['status'], 'extended_region', region)
            self.assertIsNone(region['point']); self.assertIsNone(region['covariance'])
            self.assertTrue(any(region['open_v_sides']))
            self.assertGreater(len(region['transverse_region_envelope']['v_samples_mm']), 10)

    def test_cross_source_duplicate_fragment_no_direction_and_uncalibrated_3d_rejected(self):
        _, tracks, _ = observations()
        origins = np.stack([p[0] for p in tracks]); directions = np.stack([p[1]-p[0] for p in tracks])
        covariance = np.tile(np.eye(4), (3, 1, 1))
        cases = []
        c = config(); c['track_source_groups'] = [1, 2, 1]; cases.append((c, directions))
        c = config(); c['track_fragment_ids'] = [0, 0, 1]; cases.append((c, directions))
        c = config(); c['physical_3d'] = False; cases.append((c, directions))
        d = directions.copy(); d[0] = 0.; cases.append((config(), d))
        for c, d in cases:
            result = vertex_region(origins, d, covariance, origins[:, 1], c)
            self.assertEqual(result['status'], 'invalid_input')
            self.assertIsNone(result['point'])

    def test_soft_first_layer_bound_and_reaction_volume_do_not_clamp(self):
        origins = np.array([[.2, 0.], [-.2, 0.]])
        directions = np.array([[-.1, 1.], [.1, 1.]])
        covariance = np.tile(np.diag([1e-6, 1e-8]), (2, 1, 1))
        c = config(2, 2); c['first_layer_half_extent_mm'] = 6.
        region = vertex_region(origins, directions, covariance, [0., 0.], c, dimension=2)
        self.assertEqual(region['status'], 'resolved', region)
        self.assertGreater(region['point'][1], 1.9)  # Not snapped to first-hit v=0.
        c['reaction_volume'] = dict(bounds_uv_mm=[[-5., 10.], [5., 100.]], source='synthetic explicit actual-volume-shaped test')
        region = vertex_region(origins, directions, covariance, [0., 0.], c, dimension=2)
        self.assertEqual(region['status'], 'outside_reaction_volume')
        self.assertFalse(region['success']); self.assertIsNone(region['point'])
        self.assertGreater(region['candidate_point'][1], 1.9)
        self.assertFalse(region['reaction_volume']['candidate_inside'])
        self.assertFalse(region['reaction_volume']['used_as_hard_constraint'])

    def test_two_free_projected_lines_cannot_gain_angle_information_from_common_vertex_alone(self):
        _, tracks, covariance = observations(2, noise=True)
        result = joint_fit(tracks, covariance, config(2, 2))
        self.assertEqual(result['status'], 'ok', result)
        np.testing.assert_allclose(result['vertex_free_theta_uv_deg'], result['vertex_joint_theta_uv_deg'], atol=1e-6)
        self.assertAlmostEqual(result['quality']['measurement_chi2'], result['free_quality']['measurement_chi2'], places=7)
        self.assertTrue(result['same_measurement_process_model'])
        self.assertEqual(result['quality']['measurement_count'], 16)
        self.assertFalse(result['independent_vertex_prior_from_current_tracks_used'])

    def test_generic_3d_common_fit_exact_closure_and_correlated_covariance(self):
        vertex, tracks, covariance = observations()
        result = joint_fit(tracks, covariance, config())
        self.assertEqual(result['status'], 'ok', result)
        np.testing.assert_allclose(result['vertex'], vertex, atol=1e-7)
        for fitted, observed in zip(result['fitted_points'], tracks):
            np.testing.assert_allclose(fitted, observed, atol=1e-8)
        self.assertEqual(len(result['joint_covariance_parameter_order']), len(result['joint_covariance']))
        self.assertTrue(np.isfinite(result['joint_covariance']).all())
        self.assertFalse(result['full_Kalman_fit'])
        self.assertTrue(np.all(np.linalg.norm(result['vertex_track_directions'], axis=1) > .999999))

    def test_joint_angular_kick_fit_handles_generic_curved_scattered_paths(self):
        vertex, tracks, covariance = observations()
        rng = np.random.default_rng(11027)
        # Construct three non-120deg paths with actual slope changes.
        for i, points in enumerate(tracks):
            points[3:, 0] += (.035+i*.005)*(points[3:, 1]-points[3, 1])
            points[4:, 2] -= .025*(points[4:, 1]-points[4, 1])
            points += rng.multivariate_normal(np.zeros(3), covariance[i][0], len(points))
        c = config(); c.update(process_noise_mode='tuned', angular_variance_rad2_per_mm=2e-5, max_nfev=100)
        result = joint_fit(tracks, covariance, c)
        self.assertIn(result['status'], ('ok', 'extended_region'), result)
        self.assertEqual(result['trajectory_model'], 'piecewise_linear_angular_kick_joint_least_squares')
        self.assertGreater(result['quality']['process_chi2'], 0.)
        self.assertGreater(result['free_quality']['process_chi2'], 0.)
        self.assertTrue(result['same_measurement_process_model'])
        self.assertGreater(np.ptp(result['local_directions'][0][:, 0]), .01)
        self.assertIn('unobserved eventwise deflection is not measured', result['vertex_direction_scope'])

    def test_track_derived_vertex_cannot_be_reintroduced_as_an_independent_measurement(self):
        _, tracks, covariance = observations()
        c = config(); c['vertex_prior_from_current_tracks'] = [0., -20., 50.]
        result = joint_fit(tracks, covariance, c)
        self.assertEqual(result['status'], 'invalid_input')
        c = config(); c['external_vertex_prior'] = dict(mean=[0., -20., 50.], covariance=np.eye(3),
            independent=False, derived_from_current_tracks=True, source='derived pair crossing')
        result = joint_fit(tracks, covariance, c)
        self.assertEqual(result['status'], 'invalid_input')

    def test_independent_external_observation_and_shared_clock_nuisance_are_explicit(self):
        vertex, tracks, covariance = observations()
        plain = joint_fit(tracks, covariance, config())
        c = config(); c['shared_w_offset'] = dict(sigma_mm=2., included_in_measurement_covariance=False,
                                                source='independent clock-calibration software fixture')
        shared = joint_fit(tracks, covariance, c)
        self.assertEqual(shared['status'], 'ok', shared)
        np.testing.assert_allclose(shared['vertex'], vertex, atol=1e-7)
        self.assertAlmostEqual(shared['vertex_covariance'][2, 2]-plain['vertex_covariance'][2, 2], 4., places=4)
        c['shared_w_offset']['included_in_measurement_covariance'] = True
        self.assertEqual(joint_fit(tracks, covariance, c)['status'], 'invalid_input')
        c = config(); c['external_vertex_prior'] = dict(mean=vertex, covariance=np.eye(3)*.01,
            independent=True, derived_from_current_tracks=False, source='independent external instrument software fixture')
        external = joint_fit(tracks, covariance, c)
        self.assertEqual(external['status'], 'ok', external)
        self.assertTrue(external['external_prior_only_in_joint'])

    def test_missing_covariance_scan_or_process_model_and_duplicate_depth_are_failures(self):
        _, tracks, covariance = observations()
        c = config(); c.pop('process_noise_mode')
        self.assertEqual(joint_fit(tracks, covariance, c)['status'], 'invalid_input')
        c = config(); c.pop('vertex_v_scan_mm')
        self.assertEqual(joint_fit(tracks, covariance, c)['status'], 'invalid_input')
        bad = copy.deepcopy(tracks); bad[0][1, 1] = bad[0][0, 1]
        self.assertEqual(joint_fit(bad, covariance, config())['status'], 'invalid_input')
        before = copy.deepcopy(tracks)
        joint_fit(tracks, covariance, config())
        for a, b in zip(tracks, before):np.testing.assert_array_equal(a, b)

    def test_numerical_convergence_does_not_accept_incompatible_common_origin(self):
        _, tracks, covariance = observations()
        tracks[2][:, 2] += 8.  # A precise third line from a different spatial origin.
        c = config(); c.update(max_measurement_reduced_chi2=10., max_total_reduced_chi2=10.)
        result = joint_fit(tracks, covariance, c)
        self.assertEqual(result['status'], 'incompatible_common_origin', result)
        self.assertTrue(result['quality']['solver_converged'])
        self.assertTrue(result['quality_gate_applied'])
        self.assertFalse(result['quality_gate_passed'])
        self.assertIsNone(result['vertex'])
        self.assertTrue(np.isfinite(result['candidate_vertex']).all())
        self.assertGreater(result['quality']['measurement_reduced_chi2'], 10.)

    def test_actual_reaction_volume_is_an_acceptance_guard_not_a_vertex_clamp(self):
        vertex, tracks, covariance = observations(2)
        c = config(2, 2)
        c['reaction_volume'] = dict(bounds_uv_mm=[[-50., 0.], [50., 140.]],
                                   source='synthetic explicit instrument-volume fixture')
        result = joint_fit(tracks, covariance, c)
        self.assertEqual(result['status'], 'outside_reaction_volume', result)
        self.assertFalse(result['success']); self.assertIsNone(result['vertex'])
        np.testing.assert_allclose(result['candidate_vertex'], vertex, atol=1e-7)
        np.testing.assert_allclose(result['vertex_free_theta_uv_deg'],
                                   result['vertex_joint_theta_uv_deg'], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
