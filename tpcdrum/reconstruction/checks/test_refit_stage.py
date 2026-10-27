"""Conditional refit closure on synthetic ROOT-shaped energy-stage records.

The imported tables/receipts are SOFTWARE fixtures, not real Geant4 transport.
"""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reconstruction.checks.test_physics_stages import fixture, material_fixture
from reconstruction.physics_stages import import_material_paths_event, energy_event
from reconstruction.refit_stage import refitting_event
from reconstruction.stage_io import StageWriter, read_stage


class RefitStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        record, diag, cfg = material_fixture(self.folder, *fixture())
        imported, diag = import_material_paths_event(record, diag, cfg)
        row = dict(record, **imported)
        energies, diag = energy_event(row, diag, {})
        self.assertEqual(diag['energy']['status'], 'ok', diag['energy'])
        row.update(energies)
        k = len(row['fit_ok'])
        row['fit_measurement_points_uvw_mm'] = row['fit_points_uvw_mm'].copy()
        row['fit_measurement_covariance_mm2'] = np.full((k, 12, 3, 3), np.nan)
        for track in range(k):
            for layer in np.flatnonzero(row['fit_valid_layer'][track]):
                row['fit_measurement_covariance_mm2'][track, layer] = np.diag([.04, 0., .04])
        slopes = row['fit_states'][:, 0, 2:4]
        directions = np.c_[slopes[:, 0], np.ones(k), slopes[:, 1]]
        row['track_directions_uvw'] = directions/np.linalg.norm(directions, axis=1)[:, None]
        row['track_origins_uvw_mm'] = row['fit_points_uvw_mm'][:, 0].copy()
        row['kinematics_ok'] = np.int64(1)
        row['invariant_mass_mev'] = np.float64(11182.)
        row['excitation_mev'] = np.float64(7.654)
        row['com_opening_angles_deg'] = np.array([110., 120., 130.])
        self.record, self.diagnostic = row, diag
        self.config = dict(refitting=dict(enabled=True, method='ekf_rts',
            process_noise_mode='none', material_name='material_gas',
            initial_covariance=np.diag([100., 100., .01, .01]).tolist(),
            straggling_variance_mev2_per_mm=0., minimum_layers=3))

    def test_uses_first_observed_energy_never_vertex_energy(self):
        self.record['track_vertex_energy_mev'][:] = 9999.
        prior = self.record['track_first_point_energy_mev'].copy()
        updates, diag = refitting_event(self.record, self.diagnostic, self.config)
        self.assertEqual(diag['refitting']['status'], 'ok', diag['refitting'])
        np.testing.assert_array_equal(updates['refit_ok'], [1, 1, 1])
        np.testing.assert_allclose(updates['refit_prior_energy_mev'], prior)
        np.testing.assert_allclose(updates['fit_energy_first_point_mev'], prior, atol=1e-9)
        np.testing.assert_allclose(updates['refit_input_reference_v_mm'], 0.)
        self.assertIn('NOT production vertex', diag['refitting']['prior_reference'])
        self.assertFalse(diag['refitting']['physics_accuracy_evaluated'])

    def test_any_success_invalidates_all_downstream_physics_and_snapshots_old(self):
        self.record['kinematics_candidate_valid']=np.int64(1)
        self.record['kinematics_method_accepted']=np.int64(1)
        self.record['kinematics_finding_assessment_available']=np.int64(1)
        self.record['kinematics_finding_ambiguous']=np.int64(0)
        self.record['kinematics_covariance_propagated']=np.int64(1)
        self.record['vertex_direction_valid']=np.ones(3,dtype=np.uint8)
        self.record['track_vertex_directions_uvw']=np.tile([0.,1.,0.],(3,1))
        updates, diag = refitting_event(self.record, self.diagnostic, self.config)
        self.assertTrue(updates['refit_requires_downstream_recompute'])
        self.assertEqual(updates['refit_revision'], 1)
        for flag in ('vertex_valid', 'si_association_unique', 'si_association_geometry_unique',
                     'material_request_valid', 'material_paths_embedded_valid', 'kinematics_ok',
                     'kinematics_candidate_valid','kinematics_covariance_propagated',
                     'kinematics_method_accepted','kinematics_finding_assessment_available'):
            self.assertFalse(updates[flag], flag)
        self.assertEqual(updates['kinematics_finding_ambiguous'],-1)
        self.assertFalse(updates['energy_ok'].any())
        self.assertFalse(updates['vertex_direction_valid'].any())
        self.assertTrue(np.isnan(updates['track_vertex_directions_uvw']).all())
        self.assertTrue(np.isnan(updates['track_vertex_energy_mev']).all())
        self.assertTrue(np.isnan(updates['track_first_point_energy_mev']).all())
        self.assertTrue(np.isnan(updates['vertex_uvw_mm']).all())
        self.assertTrue(np.isnan(updates['com_opening_angles_deg']).all())
        self.assertEqual(len(updates['material_path_length_mm']), 0)
        np.testing.assert_array_equal(updates['initial_fit_states'], self.record['fit_states'])
        np.testing.assert_array_equal(updates['initial_track_first_point_energy_mev'],
                                      self.record['track_first_point_energy_mev'])
        self.assertTrue(updates['initial_material_paths_embedded_valid'])
        self.assertTrue(diag['fitting']['physical_3d'])
        for section in ('vertex', 'association', 'material_request', 'material_import', 'energy', 'kinematics'):
            self.assertEqual(diag[section]['status'], 'invalidated_by_refit')

    def test_disabled_or_missing_inputs_do_not_replace_the_current_geometry(self):
        disabled, diag = refitting_event(self.record, self.diagnostic, {})
        self.assertEqual(diag['refitting']['status'], 'disabled')
        self.assertFalse(disabled['refit_requested'])
        np.testing.assert_array_equal(disabled['fit_states'], self.record['fit_states'])
        self.assertTrue(disabled['energy_ok'].all())
        self.assertTrue(disabled['vertex_valid'])
        config = copy.deepcopy(self.config); config['refitting'].pop('initial_covariance')
        missing, diag = refitting_event(self.record, self.diagnostic, config)
        self.assertFalse(missing['refit_ok'].any())
        self.assertTrue(missing['energy_ok'].all())
        self.assertFalse(missing['refit_requires_downstream_recompute'])
        self.assertEqual(set(disabled), set(missing))
        self.assertIn('initial_covariance', diag['refitting']['tracks'][0]['message'])

    def test_embedded_tables_and_root_record_are_sufficient_without_sidecars(self):
        root = self.folder/'energy_stage.root'
        with StageWriter(root, 'energy') as writer:
            writer.write(self.record, self.diagnostic)
        for pattern in ('*.json', '*.csv'):
            for file in self.folder.glob(pattern):file.unlink()
        row, diagnostic = next(read_stage(root))
        # Stale upstream file path must not be opened instead of ROOT payload.
        config = copy.deepcopy(self.config)
        config['energy'] = dict(stopping_tables='/does/not/exist/old_export.json')
        updates, diag = refitting_event(row, diagnostic, config)
        self.assertEqual(diag['refitting']['status'], 'ok', diag['refitting'])
        self.assertEqual(diag['refitting']['stopping_table_source'], 'embedded_ROOT_stopping_payload')
        self.assertTrue(updates['refit_ok'].all())
        # Calibration properties survive the merged record; old path validity does not.
        merged = dict(row, **updates)
        self.assertTrue(merged['stopping_payload_valid'])
        self.assertFalse(merged['material_paths_embedded_valid'])

    def test_stale_energy_flags_and_wrong_reference_plane_are_rejected(self):
        row = copy.deepcopy(self.record); row['energy_ok'][:] = 0
        updates, diag = refitting_event(row, self.diagnostic, self.config)
        self.assertFalse(updates['refit_ok'].any())
        row = copy.deepcopy(self.record); row['track_first_fit_point_uvw_mm'][0, 1] += 12.
        updates, diag = refitting_event(row, self.diagnostic, self.config)
        np.testing.assert_array_equal(updates['refit_ok'], [0, 1, 1])
        self.assertIn('references disagree', diag['refitting']['tracks'][0]['message'])
        self.assertFalse(updates['energy_ok'].any())

    def test_nonhomogeneous_known_material_is_not_silently_replaced_by_gas(self):
        row = copy.deepcopy(self.record)
        mask = (row['material_path_track'] == 0) & (row['material_path_segment_id'] == 1)
        self.assertTrue(mask.any())
        row['material_path_table_index'][mask] = 1  # material_foil in this synthetic export.
        config = copy.deepcopy(self.config)
        config['refitting']['homogeneous_material_between_points'] = True
        updates, diag = refitting_event(row, self.diagnostic, config)
        self.assertEqual(updates['refit_ok'][0], 0)
        self.assertIn('multiple/different materials', diag['refitting']['tracks'][0]['message'])

    def test_no_posterior_track_is_reused_as_an_observed_point(self):
        row = copy.deepcopy(self.record); row.pop('fit_measurement_points_uvw_mm')
        updates, diag = refitting_event(row, self.diagnostic, self.config)
        self.assertFalse(updates['refit_ok'].any())
        self.assertIn('fit_measurement_points_uvw_mm', diag['refitting']['message'])
        # Existing CNN/Kalman states remain stored but cannot be a new independent hit.
        np.testing.assert_array_equal(updates['fit_states'], self.record['fit_states'])

    def test_first_baseline_snapshots_survive_iterations_and_source_is_immutable(self):
        row = copy.deepcopy(self.record)
        row['initial_track_vertex_energy_mev'] = np.array([1., 2., 3.])
        before = copy.deepcopy(row)
        updates, diag = refitting_event(row, self.diagnostic, self.config)
        np.testing.assert_array_equal(updates['initial_track_vertex_energy_mev'], [1., 2., 3.])
        for key in row:
            np.testing.assert_equal(row[key], before[key])
        # A second call cannot recycle the previous prior after invalidating energy.
        repeated, repeated_diag = refitting_event(dict(row, **updates), diag, self.config)
        self.assertFalse(repeated['refit_ok'].any())
        self.assertFalse(repeated['refit_attempted'])

    def test_config_cannot_override_with_vertex_or_unrelated_energy(self):
        config = copy.deepcopy(self.config); config['refitting']['energy_first_point_mev'] = 123.
        updates, diag = refitting_event(self.record, self.diagnostic, config)
        self.assertFalse(updates['refit_ok'].any())
        self.assertIn('not config', diag['refitting']['tracks'][0]['message'])


if __name__ == '__main__':
    unittest.main()
