import tempfile
from pathlib import Path
import unittest
import numpy as np

from reconstruction.stage_io import StageWriter,read_stage,manifest
from reconstruction.workflow import run_stage,_kinematics_counts


class WorkflowHandoffChecks(unittest.TestCase):
    def test_empty_found_track_set_remains_a_failure_after_root_handoff(self):
        # A failed finder can legitimately retain an event with zero tracks.
        # Empty fit_ok must not become an all-tracks-success event via all([]).
        import uproot
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);source=directory/'finding.root'
            q=np.zeros((12,64,512),np.float32)
            q[0,20,420]=100.
            with StageWriter(source,'finding') as writer:
                writer.write(dict(entry=np.int64(5),file_index=np.int64(0),
                    source_group=np.int64(123),positive_waveform=q,
                    components=np.empty((0,12,64,512),np.float32),
                    background_waveform=q.copy(),finding_ambiguous=np.int64(1)),
                    dict(finding=dict(status='evaluation_method_failed',found_count=0,ambiguous=True)))
            config=dict(fitting=dict(minimum_layers=3),
                        vertex=dict(method='joint_tracks',joint=dict(dimension=2)))
            fitted=directory/'fitting.root';vertex=directory/'vertex.root'
            run_stage('fitting',source,fitted,config)
            source.unlink()
            row,diagnostic=next(read_stage(fitted))
            self.assertEqual(row['fit_ok'].size,0)
            self.assertEqual(diagnostic['fitting']['successful_tracks'],0)
            with uproot.open(fitted) as root:
                metrics=root['metrics'].arrays(library='np')
                values=dict(zip(metrics['name'],metrics['value']))
                self.assertEqual(values['events'],1.)
                self.assertEqual(values['stage_output_valid_fraction'],0.)
            run_stage('vertex',fitted,vertex,config)
            fitted.unlink()
            row,diagnostic=next(read_stage(vertex))
            self.assertFalse(row['vertex_valid'])
            self.assertFalse(row['vertex_projected_valid'])
            self.assertEqual(diagnostic['vertex']['status'],'fewer_than_two_measured_tracks')
            self.assertEqual(row['components'].shape,(0,12,64,512))
            self.assertTrue(np.array_equal(row['positive_waveform'],q))

    def test_saved_candidate_unknown_finding_is_never_counted_or_scored_as_accepted(self):
        # Synthetic software closure through saved energy -> kinematics ROOT ->
        # final run-summary counts and paired evaluation; no detector claim.
        import uproot
        from reconstruction.algorithms.physics import boost
        from reconstruction.evaluation import evaluate_stage
        mass=3727.3794066
        p=np.array([[10.,0.,0.],[0.,20.,0.],[-10.,-20.,0.]])
        com=np.column_stack([p,np.sqrt(mass**2+(p*p).sum(1))])
        ground=float(com[:,3].sum()-7.654);lab=boost(com,[0.,.08,.01])
        directions=lab[:,:3]/np.linalg.norm(lab[:,:3],axis=1)[:,None]
        components=np.zeros((3,12,64,512),np.float32)
        for i in range(3):components[i,:,20+3*i,420+i]=100.
        positive=components.sum(0);background=np.zeros_like(positive)
        key=dict(entry=np.int64(11),file_index=np.int64(0),source_group=np.int64(777))
        meta=dict(source_dataset_uuid='synthetic-strict-method-acceptance',source_group_partition_verified=True)
        config=dict(masses=dict(alpha_mev=mass,carbon12_ground_mev=ground),
                    kinematics=dict(acknowledge_first_point_direction_as_vertex_approximation=True))
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);energy=directory/'energy.root';output=directory/'kinematics.root'
            row=dict(key,track_vertex_energy_mev=lab[:,3]-mass,energy_ok=np.ones(3,np.uint8),
                     fit_ok=np.ones(3,np.uint8),track_directions_uvw=directions,
                     positive_waveform=positive,components=components,background_waveform=background)
            with StageWriter(energy,'energy',meta) as writer:
                writer.write(row,dict(fitting=dict(physical_3d=True)))
            run_stage('kinematics',energy,output,config)
            energy.unlink()
            candidate,diagnostics=next(read_stage(output))
            self.assertTrue(candidate['kinematics_candidate_valid'])
            self.assertTrue(candidate['kinematics_direction_model_compatible'])
            self.assertFalse(candidate['kinematics_ok']);self.assertFalse(candidate['kinematics_method_accepted'])
            self.assertTrue(np.isfinite(candidate['alpha_four_vectors_mev']).all())
            counts=_kinematics_counts(output)
            self.assertEqual(counts['accepted_events'],0);self.assertEqual(counts['valid_kinematics'],0)
            self.assertEqual(counts['candidate_kinematics_events'],1)
            self.assertEqual(counts['direction_model_compatible_events'],1)
            self.assertFalse(counts['all_events_have_physical_result'])
            with uproot.open(output) as source:
                metrics=source['metrics'].arrays(library='np')
                measured=dict(zip(metrics['name'],metrics['value']))
                self.assertEqual(measured['stage_output_valid_fraction'],0.)
                self.assertEqual(measured['kinematics_candidate_event_fraction'],1.)
                self.assertNotIn('performance/accepted_method_com_angle_deg',source)
                self.assertAlmostEqual(source['performance/candidate_com_opening_angle_deg'].values().sum(),3.)
            truth=directory/'truth.root'
            with StageWriter(truth,'truth',dict(meta,role='truth')) as writer:
                writer.write(dict(key,truth_components=components,truth_background=background,
                    truth_four_vectors_mev=lab,truth_direction_uvw=directions,truth_kinetic_mev=lab[:,3]-mass))
            result=evaluate_stage(output,truth,directory/'evaluation.root',config=config)
            self.assertEqual(result['summary']['kinematics_accepted_events'],0)
            self.assertEqual(result['summary']['kinematics_candidate_events'],1)
            evaluated,_=next(read_stage(directory/'evaluation.root',allow_truth=True))
            self.assertTrue(np.isnan(evaluated['invariant_mass_error_mev']))
            self.assertAlmostEqual(evaluated['candidate_invariant_mass_error_mev'],0.,places=8)
            # A legacy file's old success bit cannot bypass the new saved-ROOT count.
            old=dict(candidate);old.pop('kinematics_method_accepted');old['kinematics_ok']=np.int64(1)
            with StageWriter(directory/'legacy.root','kinematics',meta) as writer:writer.write(old,diagnostics)
            legacy_counts=_kinematics_counts(directory/'legacy.root')
            self.assertEqual(legacy_counts['accepted_events'],0)
            self.assertEqual(legacy_counts['candidate_kinematics_events'],1)
            self.assertFalse(legacy_counts['all_events_have_physical_result'])

    def test_missing_calibration_is_saved_without_truth_or_parent_file_dependency(self):
        # A quiet event is also a real production edge case: do not force K lines,
        # invent time-to-distance calibration, or silently promote NaN to zero.
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);previous=directory/'observed.root'
            with StageWriter(previous,'ingest') as writer:
                writer.write(dict(entry=np.int64(7),file_index=np.int64(0),
                    source_group=np.int64(77),raw_waveform=np.zeros((12,64,512),np.float32),
                    si_edep_mev=np.empty(0),has_observed_si_edep=np.uint8(0)),
                    {'observed_only':True})
                # The order previously triggered an Uproot recursive-membership bug.
                writer.histogram('quiet_adc',[0.]);writer.metrics({'events':1.})
            config={'input_roots':['/missing/original.root'],'n_tracks':3,
                    'finding':{'k':3,'method':'hough','tb_bin':4},'drift':{},'masses':{}}
            for stage in ('prepare','finding','fitting','vertex','association','material_request','material_import','energy','kinematics'):
                output=directory/(stage+'.root')
                run_stage(stage,previous,output,config)
                previous.unlink()  # The new file must be sufficient for the next task.
                previous=output
            rows=list(read_stage(previous));self.assertEqual(len(rows),1)
            arrays,diagnostics=rows[0]
            self.assertFalse(arrays['kinematics_ok'])
            self.assertFalse(np.any(arrays['fit_ok']))
            self.assertTrue(np.isnan(arrays['invariant_mass_mev']))
            self.assertIn('missing_calibration',diagnostics['fitting']['status'])
            self.assertEqual(manifest(previous)['stage'],'kinematics')
            self.assertEqual(arrays['raw_waveform'].shape,(12,64,512))
            self.assertFalse(any(name.startswith('truth_') for name in arrays))


if __name__=='__main__':unittest.main()
