"""Finding handoff integration on generated ROOT fixtures; no user data is read."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

import reconstruction.training_finding as training
from reconstruction.finding_handoff import FindingHandoff
from reconstruction.stage_io import StageWriter,read_stage,manifest
from reconstruction.workflow import run_stage
from reconstruction.checks.test_finding_handoff import components,finding,record


class FindingHandoffIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expected=Path(__file__).resolve().parents[1]/'training_finding.py'
        if Path(training.__file__).resolve()!=expected:
            raise AssertionError('Tests MUST import their own copied reconstruction package: '+str(training.__file__))
        cls.package=expected.parent

    def fixtures(self,folder):
        obs,truth=folder/'original_observed.root',folder/'original_truth.root'
        a,b=components(3),components(1);zero=np.empty((0,12,64,512),np.float32)
        qs=[a.sum(0),b.sum(0),np.zeros((12,64,512),np.float32),a.sum(0)*2.]
        references=[a[[2,0,1]],np.concatenate([np.zeros_like(b),b,np.zeros_like(b)]),
                    np.zeros((3,12,64,512),np.float32),a[[2,0,1]]*2.]
        directions=np.array([[0.,1.,0.],[.1,1.,0.],[-.1,1.,0.]])
        meta=dict(source_dataset_uuid='copied-handoff-integration-fixture',source_group_partition_verified=True)
        with StageWriter(obs,'prepare',meta) as ow,StageWriter(truth,'truth',dict(meta,role='truth',tb_bin=4)) as tw:
            for i,q in enumerate(qs):
                r=record(i,q);r.update(si_edep_mev=np.array([1.,2.]),has_observed_si_edep=np.int64(1),MCTrack=np.array([999.]))
                ow.write(r,dict(si_detector_names=['SiDet_RU','SiDet_U'],private_truth_diagnostic='must not enter sink'))
                key={k:r[k] for k in ('file_index','entry','source_group')}
                tw.write(dict(key,truth_components=references[i],truth_background=np.zeros_like(q),truth_direction_uvw=directions))
        split=folder/'split.json';split.write_text(json.dumps(dict(train=[],validation=[],test=[
            dict(file_index=0,entry=i,source_group=100+i) for i in range(4)])))
        checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'synthetic fixture, never loaded as a model')
        config=dict(methods=['hough','sequential_vertex'],tb_bin=4,finding=dict(k=3,tb_bin=4,method='sequential_vertex'))
        return dict(observations_root=str(obs),truth_root=str(truth),source_dataset_uuid=meta['source_dataset_uuid']),split,checkpoint,config,qs,a

    def observed_prediction(self,q,reverse=False):
        # Fixed observation-only pulse islands define this deterministic model
        # mock; neither reference labels nor event identifiers are arguments.
        self.assertIsInstance(q,np.ndarray);self.assertEqual(q.shape,(12,64,512))
        if float(q.sum())>2000.:raise RuntimeError('synthetic inference failure')
        parts=[]
        for tb in (120,124,128):
            if q[...,tb].sum()>0:
                part=np.zeros_like(q);part[...,tb]=q[...,tb];parts.append(part)
        comp=np.stack(parts) if parts else np.empty((0,12,64,512),np.float32)
        if reverse and len(comp)>1:comp=comp[[1,2,0]]
        fresh=finding(comp)
        fresh['diagnostics']['method']='synthetic observation-only mock'
        return fresh

    def test_single_predictions_feed_truth_free_original_slot_roots_then_uv_vertex(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);inputs,split,checkpoint,config,qs,a=self.fixtures(folder)
            outputs={method:folder/'handoff'/str(method+'.root') for method in config['methods']}
            evaluated,report=folder/'evaluation.root',folder/'reports'/'evaluation.json'
            calls=[];order=[]
            def standard(q,cfg):
                self.assertEqual(cfg['method'],'hough');calls.append(('hough',float(q.sum())));order.append('infer')
                return self.observed_prediction(q)
            def sequential(r,cfg):
                self.assertEqual(set(r),{'positive_waveform','tb_ns'})
                self.assertEqual(cfg['method'],'sequential_vertex');calls.append(('sequential_vertex',float(r['positive_waveform'].sum())));order.append('infer')
                return self.observed_prediction(r['positive_waveform'],True)
            actual_write=FindingHandoff.write;actual_metric=training.separation_metrics
            def sink_write(sink,*args,**kwargs):
                order.append('sink')
                self.assertIn('si_detector_names',kwargs['observation_diagnostic'])
                if kwargs.get('found') is not None:
                    self.assertNotIn('slot_permutation',kwargs['found'])
                    self.assertNotIn('evaluation_slot_alignment',kwargs['found']['diagnostics'])
                return actual_write(sink,*args,**kwargs)
            def score(*args,**kwargs):order.append('metric');return actual_metric(*args,**kwargs)
            with patch('reconstruction.algorithms.finding.find_event',side_effect=standard),patch(
                    'reconstruction.algorithms.sequential_finding.find_event',side_effect=sequential),patch.object(
                    FindingHandoff,'write',new=sink_write),patch.object(training,'separation_metrics',side_effect=score):
                result=training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,report,config,
                                                           observation_outputs=outputs)
            self.assertEqual(len(calls),8)  # 4 frozen cases x 2 methods, never repeated for handoff.
            self.assertEqual(order[:3],['infer','sink','metric'])
            for method in config['methods']:
                self.assertEqual(result['summary'][method]['N_events'],4)
                self.assertEqual(result['summary'][method]['zero_charge_events'],1)
                self.assertEqual(result['summary'][method]['method_failed_events'],1)
                self.assertEqual(result['summary'][method]['unscored_active_events'],1)
            self.assertEqual(result['per_event'][0]['metrics']['hough']['slot_permutation'],[2,0,1])
            # Remove ALL original inputs; both evaluation and observation-stage
            # handoff remain independently usable ROOT artifacts.
            for path in (Path(inputs['observations_root']),Path(inputs['truth_root']),split,checkpoint):path.unlink()
            evaluation_rows=list(read_stage(evaluated,allow_truth=True))
            self.assertTrue(np.isfinite(evaluation_rows[0][0]['reference_initial_theta_pad_deg']).all())
            for method,path in outputs.items():
                rows=list(read_stage(path));self.assertEqual(len(rows),4)
                self.assertEqual([r['components'].shape[0] for r,_ in rows],[3,1,0,0])
                expected=a if method=='hough' else a[[1,2,0]]
                np.testing.assert_array_equal(rows[0][0]['components'],expected)
                for i,(r,diag) in enumerate(rows):
                    np.testing.assert_array_equal(r['positive_waveform'],qs[i])
                    np.testing.assert_allclose(r['components'].sum(0)+r['background_waveform'],qs[i],atol=2.e-5)
                    np.testing.assert_allclose(r['component_fractions'].sum(0),1.,atol=2.e-5)
                    self.assertEqual(diag['si_detector_names'],['SiDet_RU','SiDet_U'])
                    self.assertNotIn('private_truth_diagnostic',diag)
                    self.assertNotIn('slot_permutation',r)
                    self.assertFalse(any(k.startswith(('truth_','reference_')) for k in r))
                self.assertTrue(rows[2][0]['finding_prediction_available'])
                self.assertTrue(rows[2][0]['finding_observed_zero_charge'])
                self.assertFalse(rows[3][0]['finding_prediction_available'])
                self.assertTrue(rows[3][0]['finding_failed'])
                metadata=manifest(path)
                self.assertEqual(metadata['role'],'observations')
                self.assertEqual(metadata['inference_config']['finding']['method'],method)
                self.assertEqual(metadata['checkpoint_sha256'],result['metadata']['checkpoint_sha256'])
                self.assertEqual(metadata['split_manifest_sha256'],result['metadata']['split_manifest_sha256'])
            flow=dict(drift={},fitting=dict(method='linear_rts',process_noise_mode='none',minimum_layers=3,minimum_layer_peak_adc=40.),
                vertex=dict(method='joint_tracks',joint=dict(dimension=2,process_noise_mode='none',minimum_layers=3,seed_layers=3,
                    vertex_v_scan_mm=[-100.,20.],first_layer_half_extent_mm=6.,max_measurement_reduced_chi2=10.)))
            fitted,vertex=folder/'fit.root',folder/'vertex.root'
            run_stage('fitting',outputs['sequential_vertex'],fitted,flow)
            outputs['sequential_vertex'].unlink()
            run_stage('vertex',fitted,vertex,flow);fitted.unlink()
            rows=list(read_stage(vertex));self.assertEqual(len(rows),4)
            self.assertTrue(rows[0][0]['vertex_projected_valid'])
            np.testing.assert_allclose(rows[0][0]['vertex_projected_uv_mm'],[0.,-12.],atol=1.e-6)
            for r,_ in rows[1:]:self.assertFalse(r['vertex_projected_valid'])
            for r,_ in rows:self.assertFalse(r['vertex_valid'])
            self.assertEqual(rows[2][0]['fit_states'].shape,(0,12,5))
            self.assertEqual(rows[3][0]['fit_states'].shape,(0,12,5))

    def test_all_path_role_collisions_and_existing_outputs_fail_before_root_reads(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);inputs,split,checkpoint,config,_,_=self.fixtures(folder)
            evaluated,report=folder/'new'/'evaluation.root',folder/'new'/'evaluation.json'
            good={method:folder/'new'/str(method+'.root') for method in config['methods']}
            for forbidden in [Path(inputs['observations_root']),Path(inputs['truth_root']),split,checkpoint,evaluated,report,good['sequential_vertex']]:
                with self.subTest(forbidden=forbidden),patch('reconstruction.stage_io.manifest',side_effect=AssertionError('ROOT inputs were opened')):
                    bad=dict(good,hough=forbidden)
                    with self.assertRaises(ValueError):training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,report,config,observation_outputs=bad)
            with self.assertRaises(ValueError):training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,report,config,observation_outputs={'hough':good['hough']})
            self.assertFalse((folder/'new').exists())
            existing=folder/'existing.root';existing.write_bytes(b'preserve existing')
            with patch('reconstruction.stage_io.manifest',side_effect=AssertionError('ROOT inputs were opened')):
                with self.assertRaises(FileExistsError):training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,report,config,
                    observation_outputs=dict(good,hough=existing))
            self.assertEqual(existing.read_bytes(),b'preserve existing')
            with self.assertRaises(ValueError):training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,evaluated,config)

    def test_real_sink_io_error_is_not_converted_to_a_scored_method_failure(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);inputs,split,checkpoint,config,_,_=self.fixtures(folder)
            outputs={method:folder/'new'/str(method+'.root') for method in config['methods']}
            evaluated,report=folder/'new'/'evaluation.root',folder/'new'/'evaluation.json'
            calls=[]
            def standard(q,cfg):calls.append(q.shape);return self.observed_prediction(q)
            with patch('reconstruction.algorithms.finding.find_event',side_effect=standard),patch.object(
                    FindingHandoff,'write',side_effect=OSError('synthetic full disk')),patch.object(
                    training,'separation_metrics',side_effect=AssertionError('Truth scoring preceded fresh handoff')):
                with self.assertRaisesRegex(OSError,'synthetic full disk'):
                    training.evaluate_finding_comparison(inputs,split,checkpoint,evaluated,report,config,observation_outputs=outputs)
            self.assertEqual(len(calls),1)
            self.assertFalse(evaluated.exists());self.assertFalse(report.exists())
            self.assertTrue(all(not path.exists() for path in outputs.values()))



if __name__=='__main__':unittest.main(verbosity=2)
