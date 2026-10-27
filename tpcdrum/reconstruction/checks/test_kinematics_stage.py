"""Synthetic numerical kinematics/direction-reference tests, not detector precision."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruction.algorithms.physics import boost
from reconstruction.kinematics_stage import kinematics_event


def fixture():
    mass = 3727.3794066
    momentum = np.array([[10., 0., 0.], [0., 20., 0.], [-10., -20., 0.]])
    com = np.column_stack([momentum, np.sqrt(mass**2+(momentum**2).sum(axis=1))])
    ground = float(com[:, 3].sum()-7.654)
    lab = boost(com, [0., .08, .01])
    directions = lab[:, :3]/np.linalg.norm(lab[:, :3], axis=1)[:, None]
    metadata = dict(schema='tpcdrum-geant4-material-export-v1',
                    calibration_sha256='SYNTHETIC_SOFTWARE_FIXTURE',
                    masses_mev=dict(alpha=mass, carbon12_ground=ground))
    record = dict(track_vertex_energy_mev=lab[:, 3]-mass,
                  energy_ok=np.ones(3, np.uint8), fit_ok=np.ones(3, np.uint8),
                  track_directions_uvw=directions,
                  stopping_metadata_json_utf8=np.frombuffer(json.dumps(metadata).encode(), np.uint8))
    return record, dict(fitting=dict(physical_3d=True)), ground


class KinematicsStageTests(unittest.TestCase):
    def test_embedded_masses_generic_angles_and_excitation(self):
        record, diag, ground = fixture()
        values, diag = kinematics_event(record, diag, {})
        self.assertTrue(values['kinematics_candidate_valid'])
        self.assertFalse(values['kinematics_ok'])
        self.assertEqual(diag['kinematics']['status'], 'candidate_only_direction_reference_unacknowledged')
        self.assertAlmostEqual(values['excitation_mev'], 7.654, places=8)
        self.assertAlmostEqual(values['invariant_mass_mev'], ground+7.654, places=8)
        self.assertGreater(values['invariant_mass_mev'], 11000.)
        self.assertAlmostEqual(values['com_opening_angles_deg'][0], 90., places=8)
        self.assertFalse(np.allclose(values['com_opening_angles_deg'], 120.))
        self.assertFalse(values['kinematics_120_degree_constraint_used'])
        self.assertFalse(values['kinematics_covariance_propagated'])
        self.assertTrue(np.isnan(values['kinematics_output_sigma']).all())

    def test_first_observed_approximation_requires_acknowledgement(self):
        record, diag, _ = fixture()
        record['finding_ambiguous']=np.int64(0)
        cfg = dict(kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        values, diag = kinematics_event(record, diag, cfg)
        self.assertTrue(values['kinematics_ok'])
        self.assertTrue(values['kinematics_uses_first_observed_tangent'])
        self.assertTrue(values['kinematics_first_tangent_approximation_acknowledged'])
        self.assertFalse(values['kinematics_vertex_direction_provenance_valid'])
        self.assertFalse(values['kinematics_physics_accuracy_validated'])

    def test_explicit_ambiguous_finding_rejects_acceptance_without_changing_candidate_observables(self):
        record,diag,_=fixture()
        cfg=dict(kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        baseline,_=kinematics_event(record,diag,cfg)
        record['finding_ambiguous']=np.int64(1)
        rejected,notes=kinematics_event(record,diag,cfg)
        self.assertTrue(rejected['kinematics_candidate_valid'])
        self.assertFalse(rejected['kinematics_ok']);self.assertFalse(rejected['kinematics_method_accepted'])
        self.assertTrue(rejected['kinematics_finding_assessment_available'])
        self.assertEqual(rejected['kinematics_finding_ambiguous'],1)
        self.assertEqual(notes['kinematics']['status'],'candidate_only_ambiguous_finding')
        for key in ('alpha_four_vectors_mev','com_four_vectors_mev','invariant_mass_mev','excitation_mev',
                    'com_opening_angles_deg','dalitz_xy'):
            np.testing.assert_array_equal(rejected[key],baseline[key])

    def test_unknown_finding_preserves_candidates_but_never_acceptance(self):
        record,diag,_=fixture()
        cfg=dict(kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        unknown,notes=kinematics_event(record,diag,cfg)
        self.assertFalse(unknown['kinematics_ok']);self.assertTrue(unknown['kinematics_candidate_valid'])
        self.assertTrue(unknown['kinematics_direction_model_compatible'])
        self.assertFalse(unknown['kinematics_method_accepted'])
        self.assertFalse(unknown['kinematics_finding_assessment_available'])
        self.assertEqual(unknown['kinematics_finding_ambiguous'],-1)
        self.assertEqual(notes['kinematics']['finding_assessment']['status'],'unknown_not_assessed')
        self.assertEqual(notes['kinematics']['status'],'candidate_only_finding_not_assessed')
        record['finding_ambiguous']=np.int64(0)
        accepted,_=kinematics_event(record,diag,cfg)
        self.assertTrue(accepted['kinematics_ok']);self.assertTrue(accepted['kinematics_method_accepted'])
        self.assertTrue(accepted['kinematics_finding_assessment_available'])
        self.assertEqual(accepted['kinematics_finding_ambiguous'],0)

    def test_malformed_conflicting_declarations_never_accept_valid_zero(self):
        cfg=dict(kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        for invalid in (np.nan,2.,np.array([0]),np.array([0,1])):
            with self.subTest(invalid=repr(invalid)):
                record,diag,_=fixture()
                record['finding_ambiguous']=invalid;diag['finding']=dict(ambiguous=False)
                values,notes=kinematics_event(record,diag,cfg)
                self.assertTrue(values['kinematics_candidate_valid'])
                self.assertTrue(values['kinematics_direction_model_compatible'])
                self.assertFalse(values['kinematics_ok']);self.assertFalse(values['kinematics_method_accepted'])
                self.assertFalse(values['kinematics_finding_assessment_available'])
                self.assertEqual(values['kinematics_finding_ambiguous'],-1)
                self.assertEqual(notes['kinematics']['finding_assessment']['invalid_declarations'],['finding_ambiguous branch'])
                self.assertEqual(notes['kinematics']['finding_assessment']['valid_declarations'],
                                 [dict(source='finding diagnostic',ambiguous=False)])

    def test_diagnostic_ambiguity_and_conflicting_declarations_reject_acceptance(self):
        record,diag,_=fixture()
        record['finding_ambiguous']=np.int64(0);diag['finding']=dict(ambiguous=True)
        cfg=dict(kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        values,notes=kinematics_event(record,diag,cfg)
        self.assertTrue(values['kinematics_candidate_valid']);self.assertFalse(values['kinematics_ok'])
        self.assertTrue(notes['kinematics']['finding_assessment']['disagreement'])
        del record['finding_ambiguous']
        fallback,_=kinematics_event(record,diag,cfg)
        self.assertTrue(fallback['kinematics_finding_assessment_available']);self.assertFalse(fallback['kinematics_ok'])

    def test_vertex_reference_cannot_relabel_first_observed_tangent(self):
        record, diag, _ = fixture()
        cfg = dict(kinematics=dict(direction_reference='vertex'))
        values, failed = kinematics_event(record, diag, cfg)
        self.assertFalse(values['kinematics_candidate_valid'])
        self.assertIn('provenance', failed['kinematics']['message'])
        record.update(track_vertex_directions_uvw=record['track_directions_uvw'],
                      vertex_direction_valid=np.ones(3, np.uint8))
        diag['vertex_direction'] = dict(source='explicit synthetic observation-only back-transport fixture', observation_only=True)
        record['finding_ambiguous']=np.int64(0)
        values, passed = kinematics_event(record, diag, cfg)
        self.assertTrue(values['kinematics_ok'])
        self.assertTrue(values['kinematics_vertex_direction_provenance_valid'])
        self.assertFalse(values['kinematics_uses_first_observed_tangent'])

    def test_only_declared_full_joint_covariance_is_propagated(self):
        record, diag, _ = fixture()
        covariance = np.diag([1e-5, 1e-9, 1e-9]*3)
        covariance[0, 3] = covariance[3, 0] = 2e-6
        cfg = dict(kinematics=dict(full_covariance_9x9=covariance.tolist()))
        values, missing = kinematics_event(record, diag, cfg)
        self.assertFalse(values['kinematics_covariance_propagated'])
        self.assertIn('scope', missing['kinematics']['uncertainty_status'])
        cfg['kinematics']['covariance_is_full_joint'] = True
        values, passed = kinematics_event(record, diag, cfg)
        self.assertTrue(values['kinematics_covariance_propagated'])
        self.assertTrue(np.isfinite(values['kinematics_output_sigma']).all())
        self.assertGreater(values['kinematics_output_sigma'][0], 0.)
        np.testing.assert_allclose(values['kinematics_full_input_covariance'], covariance)
        cfg['kinematics']['full_covariance_9x9'] = np.eye(3).tolist()
        values, failed = kinematics_event(record, diag, cfg)
        self.assertFalse(values['kinematics_covariance_propagated'])
        self.assertEqual(failed['kinematics']['uncertainty_status'], 'invalid_explicit_covariance')

    def test_missing_masses_or_three_independent_energies_fail_closed(self):
        record, diag, _ = fixture()
        del record['stopping_metadata_json_utf8']
        a, failed = kinematics_event(record, diag, {})
        self.assertFalse(a['kinematics_candidate_valid'])
        self.assertIn('masses', failed['kinematics']['message'])
        record, diag, _ = fixture();record['energy_ok'][1] = 0
        b, failed = kinematics_event(record, diag, {})
        self.assertFalse(b['kinematics_candidate_valid'])
        self.assertEqual(set(a), set(b))


if __name__ == '__main__':
    unittest.main(verbosity=2)
