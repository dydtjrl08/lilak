"""Synthetic software closure for ROOT-ready physics handlers, not detector accuracy.

Material CSVs/receipts here are explicit synthetic fixtures. No native Geant4
tracing or transport validation is claimed by these tests.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruction.physics_stages import (associate_event, material_request_event,
    import_material_paths_event, energy_event, stable_path_id, write_material_requests)
from reconstruction.stage_io import StageWriter, read_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(shared=False):
    slopes = np.array([0., .001]) if shared else np.array([-.2, 0., .2])
    k = len(slopes)
    points, states = np.full((k, 12, 3), np.nan), np.full((k, 12, 5), np.nan)
    valid = np.zeros((k, 12), dtype=np.uint8)
    for track, slope in enumerate(slopes):
        for layer, v in enumerate([0., 12., 24.]):
            points[track, layer] = [slope*(v+10.), v, 0.]
            states[track, layer, :4] = [points[track, layer, 0], 0., slope, 0.]
            valid[track, layer] = 1
    centers = [0.] if shared else [42., -42., 0.]
    names = ['center'] if shared else ['right', 'left', 'center']
    detectors = [dict(name=name, center_uvw_mm=[u, 199., 0.],
                      axis_u_uvw=[1., 0., 0.], axis_v_uvw=[0., 0., 1.],
                      normal_uvw=[0., 1., 0.], width_mm=35., height_mm=75.,
                      thickness_mm=1., material='material_si')
                 for name, u in zip(names, centers)]
    record = dict(file_index=np.int64(1), entry=np.int64(5), fit_ok=np.ones(k, dtype=np.uint8),
                  fit_points_uvw_mm=points, fit_states=states, fit_valid_layer=valid,
                  # Deliberately WRONG global/first direction for Si projection:
                  # a handler must use the last local fitted tangent instead.
                  track_directions_uvw=np.tile([0., 1., 0.], (k, 1)),
                  vertex_valid=np.int64(1), vertex_uvw_mm=np.array([0., -10., 0.]),
                  has_observed_si_edep=np.int64(1),
                  si_edep_mev=np.array([20.]) if shared else np.array([9., 10., 8.]))
    diagnostic = dict(observed_only=True, fitting=dict(physical_3d=True),
                      observed_si_names=['center'] if shared else ['center', 'right', 'left'])
    config = dict(si_geometry=dict(detectors=detectors), energy=dict(
        association_position_sigma_mm=.1, association_gate_sigma=3.,
        si_energy_sigma_mev=[.3] if shared else [.3, .4, .2],
        alpha_stops_in_Si_validated=True, material_paths_validated=True))
    return record, diagnostic, config


def material_fixture(folder, record, diagnostic, config):
    """Create strict native-export-SHAPED synthetic data, no Geant4 execution."""
    requests, diag = material_request_event(record, diagnostic, config)
    row = dict(record, **requests)
    request_root, request_csv = folder/'request.root', folder/'request.csv'
    with StageWriter(request_root, 'material_request') as writer:
        writer.write(row, diag)
    written = write_material_requests(request_root, request_csv)
    materials = []
    for material, dedx in [('material_gas', .01), ('material_foil', 1.), ('material_si', 50.)]:
        energy = np.linspace(0., 100., 1001)
        materials.append(dict(material=material, particle_pdg=1000020040,
            energy_mev=energy.tolist(), range_mm=(energy/dedx).tolist(),
            dedx_mev_per_mm=np.full(len(energy), dedx).tolist(),
            units=dict(energy='MeV', range='mm', length='mm', dedx='MeV/mm'),
            provenance=dict(source='SYNTHETIC SOFTWARE FIXTURE; NOT NATIVE TRANSPORT', particle_pdg=1000020040)))
    material_file = folder/'materials.json'
    material_file.write_text(json.dumps(dict(schema='tpcdrum-geant4-material-export-v1',
        detectors=config['si_geometry']['detectors'], tables=materials,
        physical_accuracy_validated=False, all_material_tables_complete=True)))
    output = folder/'paths.csv'
    with request_csv.open(newline='') as source, output.open('w', newline='') as target:
        reader, writer = csv.DictReader(source), csv.writer(target, lineterminator='\n')
        writer.writerow(['path_id', 'segment_id', 'part', 'material', 'length_mm', 'volume'])
        rows = list(reader)
        max_segment = {}
        for r in rows:
            max_segment[r['path_id']] = max(max_segment.get(r['path_id'], 0), int(r['segment_id']))
        for r in rows:
            distance = np.linalg.norm(np.array([r['u1'], r['v1'], r['w1']], float)-
                                      np.array([r['u0'], r['v0'], r['w0']], float))
            if int(r['segment_id']) == max_segment[r['path_id']]:
                writer.writerow([r['path_id'], r['segment_id'], 0, 'material_gas', distance-.1, 'synthetic gas'])
                writer.writerow([r['path_id'], r['segment_id'], 1, 'material_foil', .1, 'synthetic foil'])
            elif distance > 0:
                writer.writerow([r['path_id'], r['segment_id'], 0, 'material_gas', distance, 'synthetic gas'])
    receipt = folder/'receipt.json'
    receipt.write_text(json.dumps(dict(executed=True, generated_events=0,
        reconstructed_path_sha256=written['sha256'], material_path_csv_sha256=digest(output),
        material_manifest_sha256=digest(material_file),
        test_scope='SYNTHETIC RECEIPT SHAPE ONLY; NO NATIVE EXECUTION')))
    config = dict(config, energy=dict(config['energy'], stopping_tables=str(material_file),
        material_path_csv=str(output), material_path_request_csv=str(request_csv),
        material_path_receipt=str(receipt)))
    return row, diag, config


class PhysicsStageTests(unittest.TestCase):
    def test_last_tangent_name_alignment_and_front_face(self):
        record, diag, cfg = fixture()
        values, diag = associate_event(record, diag, cfg)
        self.assertEqual(diag['association']['status'], 'ok')
        np.testing.assert_array_equal(values['track_si_detector_id'], [1, 2, 0])
        np.testing.assert_array_equal(values['track_si_observed_index'], [2, 0, 1])
        np.testing.assert_allclose(values['track_si_entrance_uvw_mm'][:, 1], 198.5)
        np.testing.assert_allclose(values['track_si_entrance_uvw_mm'][:, 0], [-41.7, 0., 41.7])
        self.assertLess(values['track_last_fit_direction_uvw'][0, 0], -.1)
        self.assertTrue(values['si_association_unique'])

    def test_stable_request_identity_and_lengths(self):
        record, diag, cfg = fixture()
        values, diag = material_request_event(record, diag, cfg)
        self.assertTrue(values['material_request_valid'])
        self.assertEqual(len(values['material_request_track']), 12)
        self.assertEqual(len(set(values['material_request_path_id'])), 3)
        self.assertEqual(values['material_request_path_id'][0], stable_path_id(1, 5, 0))
        self.assertNotEqual(stable_path_id(1, 5, 0), stable_path_id(2, 5, 0))
        self.assertNotEqual(stable_path_id(1, 5, 0), stable_path_id(1, 6, 0))
        with self.assertRaises(ValueError):stable_path_id(2**16, 0, 0)

    def test_skipping_invalid_requests_is_explicit_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record, diag, cfg = fixture()
            good, good_diag = material_request_event(record, diag, cfg)
            bad_record = dict(record, entry=np.int64(6), vertex_valid=np.int64(0))
            bad, bad_diag = material_request_event(bad_record, diag, cfg)
            source = folder/'requests.root'
            with StageWriter(source, 'material_request') as writer:
                writer.write(dict(record, **good), good_diag)
                writer.write(dict(bad_record, **bad), bad_diag)
            with self.assertRaises(ValueError):write_material_requests(source, folder/'strict.csv')
            self.assertFalse((folder/'strict.csv').exists())
            result = write_material_requests(source, folder/'selected.csv', skip_invalid=True)
            self.assertEqual(result['accepted_events'], 1)
            self.assertEqual(result['skipped_event_count'], 1)
            self.assertEqual(result['skipped_events'][0]['entry'], 6)
            self.assertTrue(result['skip_invalid_explicitly_requested'])

    def test_material_import_energy_and_root_only_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record, diag, cfg = material_fixture(folder, *fixture())
            updates, diag = import_material_paths_event(record, diag, cfg)
            self.assertEqual(diag['material_import']['status'], 'ok')
            self.assertFalse(diag['material_import']['native_geometry_verified_here'])
            self.assertTrue(updates['material_paths_embedded_valid'])
            row = dict(record, **updates)
            imported = folder/'imported.root'
            with StageWriter(imported, 'material_import') as writer:writer.write(row, diag)
            for path in folder.glob('*.json'):path.unlink()
            for path in folder.glob('*.csv'):path.unlink()
            row, diag = next(read_stage(imported))
            values, diag = energy_event(row, diag, {})
            self.assertEqual(diag['energy']['status'], 'ok', diag['energy'])
            np.testing.assert_array_equal(values['energy_ok'], [1, 1, 1])
            deposits = np.array([8., 9., 10.])
            lengths = 208.5*np.sqrt(1+np.array([-.2, 0., .2])**2)
            np.testing.assert_allclose(values['track_vertex_energy_mev'], deposits+.01*(lengths-.1)+.1)
            vertex_to_first = 10*np.sqrt(1+np.array([-.2, 0., .2])**2)
            np.testing.assert_allclose(values['track_first_point_energy_mev'], values['track_vertex_energy_mev']-.01*vertex_to_first)
            np.testing.assert_allclose(values['track_vertex_energy_sigma_mev'], [.2, .3, .4])
            self.assertFalse(diag['energy']['equal_split_imposed'])
            self.assertTrue(diag['energy']['eventwise_straggling_not_inverted'])

    def test_shared_module_never_divided_into_mc_or_equal_energies(self):
        with tempfile.TemporaryDirectory() as directory:
            record, diag, cfg = material_fixture(Path(directory), *fixture(shared=True))
            self.assertTrue(record['si_association_geometry_unique'])
            self.assertFalse(record['si_association_unique'])
            updates, diag = import_material_paths_event(record, diag, cfg)
            values, diag = energy_event(dict(record, **updates), diag, cfg)
            self.assertEqual(diag['energy']['status'], 'unresolved_energy')
            self.assertTrue(np.isnan(values['track_vertex_energy_mev']).all())
            self.assertFalse(values['energy_ok'].any())
            self.assertTrue(all(t['status'] == 'unresolved_shared_detector' for t in diag['energy']['tracks']))

    def test_receipt_hash_and_material_length_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            record, diag, cfg = material_fixture(Path(directory), *fixture())
            receipt = Path(cfg['energy']['material_path_receipt'])
            obj = json.loads(receipt.read_text());obj['reconstructed_path_sha256'] = 'wrong'
            receipt.write_text(json.dumps(obj))
            values, failed = import_material_paths_event(record, diag, cfg)
            self.assertFalse(values['material_paths_embedded_valid'])
            self.assertIn('hash', failed['material_import']['message'])
            obj['reconstructed_path_sha256'] = digest(cfg['energy']['material_path_request_csv'])
            path = Path(cfg['energy']['material_path_csv'])
            rows = path.read_text().splitlines();line = rows[1].split(',');line[4] = '9999';rows[1] = ','.join(line)
            path.write_text('\n'.join(rows)+'\n')
            obj['material_path_csv_sha256'] = digest(path);receipt.write_text(json.dumps(obj))
            values, failed = import_material_paths_event(record, diag, cfg)
            self.assertFalse(values['material_paths_embedded_valid'])
            self.assertIn('close', failed['material_import']['message'])

    def test_refit_makes_material_paths_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            record, diag, cfg = material_fixture(Path(directory), *fixture())
            updates, diag = import_material_paths_event(record, diag, cfg)
            row = dict(record, **updates)
            row['fit_states'] = row['fit_states'].copy();row['fit_states'][0, 2, 2] += .001
            values, failed = energy_event(row, diag, cfg)
            self.assertFalse(values['energy_ok'].any())
            self.assertIn('stale', failed['energy']['message'])

    def test_material_index_is_reused_and_rejects_orphan_global_piece(self):
        with tempfile.TemporaryDirectory() as directory:
            record, diag, cfg = material_fixture(Path(directory), *fixture())
            _, first = import_material_paths_event(record, diag, cfg)
            self.assertEqual(first['material_import']['status'], 'ok')
            self.assertFalse(first['material_import']['material_index_cache_hit'])
            _, second = import_material_paths_event(record, diag, cfg)
            self.assertTrue(second['material_import']['material_index_cache_hit'])
            self.assertFalse(second['material_import']['global_CSV_reparsed_for_this_event'])
            path = Path(cfg['energy']['material_path_csv'])
            with path.open('a') as stream:
                stream.write('123456789,0,0,material_gas,1,orphan outside this event\n')
            receipt_path = Path(cfg['energy']['material_path_receipt'])
            receipt = json.loads(receipt_path.read_text())
            receipt['material_path_csv_sha256'] = digest(path)
            receipt_path.write_text(json.dumps(receipt))
            values, failed = import_material_paths_event(record, diag, cfg)
            self.assertFalse(values['material_paths_embedded_valid'])
            self.assertIn('orphan', failed['material_import']['message'])

    def test_missing_calibration_schema_and_unknown_names(self):
        record, diag, cfg = fixture()
        missing, failed = associate_event(record, diag, {})
        self.assertFalse(missing['si_association_unique'])
        self.assertEqual(failed['association']['status'], 'missing_calibration')
        wrong = dict(diag, observed_si_names=['unknown', 'right', 'left'])
        values, failed = associate_event(record, wrong, cfg)
        self.assertFalse(values['si_association_unique'])
        self.assertIn('one-to-one', failed['association']['message'])
        a, _ = energy_event(record, diag, {})
        b, _ = energy_event(record, diag, cfg)
        self.assertEqual(set(a), set(b))
        self.assertFalse(a['energy_ok'].any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
