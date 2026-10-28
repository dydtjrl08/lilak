"""Analytic/synthetic software checks, not detector physics resolution claims."""
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruction.algorithms.physics import (
    StoppingTable, load_material_paths, inverse_energy, forward_energy, silicon_entrance_energy,
    associate_si, shared_si_energy_fit, four_vectors, boost, kinematics,
    propagate_kinematics_uncertainty)


def table(material='gas', power=1., scale=.5):
    energy = np.linspace(0., 100., 10001)
    ranges = scale*energy**power
    # For the synthetic range function; derivative at zero is not used in tests.
    dedx = 1/(scale*power*np.maximum(energy, .001)**(power-1))
    return StoppingTable(energy, ranges, dedx, material,
                         {'source': 'synthetic_analytic_fixture', 'particle_pdg': 1000020040,
                          'density_g_cm3': .001, 'scope': 'software_only'})


def detector(u):
    return dict(center_uvw_mm=[u, 199., 70.], normal_uvw=[0., 1., 0.],
                axis_u_uvw=[1., 0., 0.], axis_v_uvw=[0., 0., 1.],
                width_mm=40., height_mm=75.)


class EnergyTests(unittest.TestCase):
    def test_navigator_path_completeness_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'material_paths.csv'
            path.write_text('path_id,segment_id,part,material,length_mm,volume\n'
                            '17,1,0,foil,0.1,window\n'
                            '17,0,1,gas,60,tpc\n'
                            '17,0,0,gas,40,tpc\n')
            found = load_material_paths(path, {(17, 0): 100., (17, 1): .1})
            self.assertEqual([p['material'] for p in found[17]], ['gas', 'gas', 'foil'])
            with self.assertRaisesRegex(ValueError, 'close'):
                load_material_paths(path, {(17, 0): 99., (17, 1): .1})
            with self.assertRaisesRegex(ValueError, 'all submitted'):
                load_material_paths(path, {(17, 0): 100., (17, 1): .1, (18, 0): 10.})

    def test_material_path_uncertainty_is_propagated(self):
        tables = {'gas': table('gas', 1., .5), 'foil': table('foil', 1., .1)}
        path = [{'material': 'gas', 'length_mm': 2.}, {'material': 'foil', 'length_mm': .1}]
        covariance = np.array([[.01, .001], [.001, .0004]])
        inverse = inverse_energy(10., path, tables, .2, path_validated=True,
                                 path_covariance_mm2=covariance)
        np.testing.assert_allclose(inverse['path_length_jacobian_mev_per_mm'], [2., 10.])
        variance = .2**2 + np.array([2., 10.])@covariance@np.array([2., 10.])
        self.assertAlmostEqual(inverse['supplied_uncertainty_sigma_mev'], np.sqrt(variance))

    def test_material_order_forward_inverse_closure(self):
        tables = {'gas': table('gas', 1.5, 2.), 'foil': table('foil', 1.8, .01)}
        path = [{'material': 'gas', 'length_mm': 100.}, {'material': 'foil', 'length_mm': .1}]
        forward = forward_energy(30., path, tables)
        self.assertEqual(forward['status'], 'ok')
        inverse = inverse_energy(forward['energy_mev'], path, tables, .05, path_validated=True)
        self.assertEqual(inverse['status'], 'ok')
        self.assertAlmostEqual(inverse['vertex_energy_mev'], 30., places=10)
        self.assertEqual([h['material'] for h in inverse['reverse_history']], ['foil', 'gas'])
        self.assertFalse(inverse['eventwise_energy_loss_recovered_exactly'])
        self.assertGreater(inverse['boundary_only_sigma_mev'], 0.)

    def test_inverse_fails_closed_without_material_path_or_table_domain(self):
        tables = {'gas': table()}
        no_path = inverse_energy(10., [{'material': 'gas', 'length_mm': 2.}], tables)
        self.assertEqual(no_path['status'], 'unvalidated_material_path')
        self.assertIsNone(no_path['vertex_energy_mev'])
        with self.assertRaisesRegex(ValueError, 'extrapolation forbidden'):
            inverse_energy(99., [{'material': 'gas', 'length_mm': 100.}], tables,
                           path_validated=True)
        with self.assertRaisesRegex(ValueError, 'Missing stopping table'):
            inverse_energy(10., [{'material': 'unknown', 'length_mm': 1.}], tables,
                           path_validated=True)

    def test_units_monotonicity_and_provenance(self):
        with self.assertRaises(ValueError):
            StoppingTable([1., 2.], [1., 1.], [1., 1.], 'bad')
        with self.assertRaises(ValueError):
            StoppingTable([1., 2.], [1., 2.], [1., 1.], 'bad', units={})
        unproven = StoppingTable([0., 100.], [0., 50.], [2., 2.], 'gas')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            inverse_energy(10., [{'material': 'gas', 'length_mm': 1.}], {'gas': unproven},
                           path_validated=True)

    def test_deposit_not_silently_equal_to_entrance_energy(self):
        si = table('Si', 1., .5)
        unresolved = silicon_entrance_energy(8., 12., 1., si)
        self.assertEqual(unresolved['status'], 'ambiguous_stopping_or_punchthrough')
        self.assertIsNone(unresolved['energy_at_si_mev'])
        stopped = silicon_entrance_energy(8., 12., 1., si, kinetic_upper_bound_mev=20.)
        self.assertEqual(stopped['status'], 'ok')
        self.assertEqual(stopped['energy_at_si_mev'], 8.)
        inconsistent = silicon_entrance_energy(8., .1, 1., si, stopping_verified=True)
        self.assertEqual(inconsistent['status'], 'stopping_assumption_inconsistent')

    def test_three_distinct_modules_and_shared_module_unidentifiable(self):
        detectors = [detector(-50.), detector(0.), detector(50.)]
        origins = np.array([[-50., 0., 70.], [0., 0., 70.], [50., 0., 70.]])
        directions = np.array([[0., 2., 0.]]*3)
        original = directions.copy()
        found = associate_si(origins, directions, detectors, [8., 9., 10.])
        np.testing.assert_array_equal(directions, original)
        self.assertTrue(found['unique'])
        self.assertEqual(found['hypotheses'][0]['track_detector_ids'], [0, 1, 2])
        np.testing.assert_array_equal(found['hypotheses'][0]['track_deposit_mev'], [8., 9., 10.])
        shared = associate_si([[-10., 0., 70.], [0., 0., 70.], [10., 0., 70.]],
                              [[0., 1., 0.]]*3, detectors, [0., 30., 0.])
        self.assertFalse(shared['unique'])
        self.assertEqual(shared['hypotheses'][0]['status'], 'unresolved_shared_detector')
        self.assertTrue(np.isnan(shared['hypotheses'][0]['track_deposit_mev']).all())

    def test_shared_energy_fit_needs_observed_prior(self):
        no_prior = shared_si_energy_fit([30.], [[.04]], [0, 0, 0])
        self.assertEqual(no_prior['status'], 'unresolved_shared_detector')
        self.assertEqual(no_prior['unidentifiable_dimensions'], 2)
        solved = shared_si_energy_fit([30.], [[.04]], [0, 0, 0], [8., 10., 12.], np.eye(3))
        np.testing.assert_allclose(solved['track_energy_mev'], [8., 10., 12.], atol=1e-10)
        self.assertTrue(solved['prior_used'])
        self.assertFalse(solved['equal_split_imposed'])
        self.assertTrue(np.all(np.linalg.eigvalsh(solved['covariance_mev2']) >= -1e-12))
        unique = shared_si_energy_fit([8., 9., 10.], np.eye(3), [0, 1, 2])
        np.testing.assert_allclose(unique['track_energy_mev'], [8., 9., 10.])

    def test_shared_fit_honors_nonnegative_boundary(self):
        fitted = shared_si_energy_fit([.5], [[.01]], [0, 0], [.01, 10.], np.eye(2))
        self.assertTrue(np.all(fitted['track_energy_mev'] >= 0))
        self.assertEqual(fitted['track_energy_mev'][0], 0.)


class KinematicsTests(unittest.TestCase):
    mass = 3727.3794066  # Explicit synthetic-fixture mass, not a default calibration.

    def equal_fixture(self):
        angles = np.arange(3)*2*np.pi/3
        directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])
        com = four_vectors(np.full(3, .126), directions, self.mass)
        ground = float(com[:, 3].sum()-7.654)
        return com, boost(com, [.03, .08, .01]), ground

    def test_equal_energy_com_is_120_after_lab_boost(self):
        com, lab, ground = self.equal_fixture()
        reco = kinematics(lab, ground)
        np.testing.assert_allclose(reco['com_four_vectors'], com, atol=1e-10)
        np.testing.assert_allclose(reco['com_opening_angles_deg'], [120.]*3, atol=1e-8)
        np.testing.assert_allclose(reco['com_energy_fractions'], [1/3.]*3, atol=1e-10)
        np.testing.assert_allclose(reco['dalitz_xy'], [0., 0.], atol=1e-10)
        self.assertAlmostEqual(reco['excitation_mev'], 7.654, places=8)
        self.assertGreater(reco['invariant_mass_mev'], 11000.)
        self.assertFalse(reco['imposed_120_degree_constraint'])
        self.assertLess(reco['com_momentum_residual_mev'], 1e-9)

    def test_generic_three_body_not_forced_to_120(self):
        momentum = np.array([[10., 0., 0.], [0., 20., 0.], [-10., -20., 0.]])
        com = np.column_stack([momentum, np.sqrt(self.mass**2+np.sum(momentum**2, axis=1))])
        lab = boost(com, [.03, .08, .01])
        reco = kinematics(lab, 11174.)
        self.assertFalse(np.allclose(reco['com_opening_angles_deg'], 120.))
        self.assertAlmostEqual(reco['com_opening_angles_deg'][0], 90., places=8)
        np.testing.assert_allclose(reco['com_four_vectors'], com, atol=1e-10)
        np.testing.assert_allclose(reco['com_energy_fractions'].sum(), 1.)

    def test_on_shell_and_bad_input(self):
        vectors = four_vectors([1., 2., 3.], [[1., 2., 3.]]*3, self.mass)
        np.testing.assert_allclose(vectors[:, 3]**2-np.sum(vectors[:, :3]**2, axis=1),
                                   self.mass**2, rtol=1e-12)
        with self.assertRaises(ValueError):
            boost(vectors, [1., 0., 0.])
        with self.assertRaises(ValueError):
            four_vectors([-1., 1., 1.], [[1., 0., 0.]]*3, self.mass)
        with self.assertRaises(ValueError):
            kinematics(np.array([[10., 0., 0., 1.]]*3), 11174.)

    def test_correlated_delta_and_mc_uncertainty(self):
        _, lab, ground = self.equal_fixture()
        energies = lab[:, 3]-self.mass
        slopes = lab[:, [0, 2]]/lab[:, 1, None]
        covariance = np.diag([.0001, 1e-8, 1e-8]*3)
        covariance[0, 3] = covariance[3, 0] = .00002
        delta = propagate_kinematics_uncertainty(energies, slopes, covariance, self.mass, ground)
        mc = propagate_kinematics_uncertainty(energies, slopes, covariance, self.mass, ground,
                                             method='mc', samples=1200, seed=123)
        self.assertTrue(np.isfinite(delta['sigma']).all())
        self.assertTrue(np.all(delta['sigma'] > 0))
        np.testing.assert_allclose(mc['sigma'], delta['sigma'], rtol=.15)
        self.assertEqual(mc['valid_draw_fraction'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
