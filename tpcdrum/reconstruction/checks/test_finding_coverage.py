"""Fixed-cohort finding coverage; synthetic fixtures are not detector results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.training_finding import (
    separation_metrics, _finding_coverage_summary, _annotate_finding_score,
    evaluate_finding_comparison, _evaluation_fractions, _component_centroid_ols_angles)
from reconstruction.stage_io import StageWriter, read_stage


def fractions_for_observation(q,k):
    nt=128;charge=q.reshape(12,64,nt,4).sum(-1)
    p=np.zeros((k+1,12,64,nt),np.float32);p[-1]=1.
    for pad in range(min(k,3)):
        active=charge[:,pad]>0
        p[pad,:,pad][active]=1.;p[-1,:,pad][active]=0.
    return p


def base_metric(charge):
    truth=np.zeros((4,*charge.shape));truth[-1]=1.
    return separation_metrics(truth,truth,charge)


class FindingCoverageTests(unittest.TestCase):
    def test_zero_charge_is_undefined_not_perfect_or_a_clamped_denominator(self):
        q=np.zeros((12,64,2));truth=np.zeros((4,12,64,2));truth[-1]=1.
        pred=np.full_like(truth,.25)
        result=separation_metrics(pred,truth,q)
        self.assertEqual(result['positive_charge'],0.)
        self.assertEqual(result['absolute_charge_error'],0.)
        self.assertTrue(np.isfinite(result['absolute_charge_error']))
        self.assertTrue(np.isnan(result['charge_l1']))
        self.assertFalse(result['charge_l1_defined'])
        self.assertTrue(result['zero_charge'])
        self.assertFalse(result['true_visible'].any());self.assertFalse(result['recovered'].any())
        self.assertTrue(np.isnan(result['precision']).all())
        declared=_annotate_finding_score(result,dict(status='hypothesis_supported_NOT_physics_validated',ambiguous=False),True)
        self.assertFalse(declared['method_declaration_success'])

    def test_nonzero_l1_math_remains_exact_and_low_visible_stays_in_cohort(self):
        q=np.zeros((12,64,2));q[0,0,0]=10.
        truth=np.zeros((2,12,64,2));truth[0]=1.
        wrong=truth[::-1].copy()
        metric=separation_metrics(wrong,truth,q)
        self.assertEqual(metric['positive_charge'],10.)
        self.assertEqual(metric['absolute_charge_error'],20.)
        self.assertEqual(metric['charge_l1'],2.)
        self.assertTrue(metric['charge_l1_defined'])
        self.assertEqual(metric['true_visible_count'],0)
        self.assertTrue(metric['nonzero_low_visible'])

    def test_pooled_all_zero_and_conditional_active_denominators(self):
        zero=base_metric(np.zeros((12,64,1)))
        summary=_finding_coverage_summary([zero,copy.deepcopy(zero)],3)
        self.assertIsNone(summary['pooled_charge_l1'])
        self.assertIsNone(summary['mean_active_event_charge_l1'])
        self.assertEqual(summary['pooled_absolute_charge_error'],0.)
        self.assertEqual(summary['pooled_scored_positive_charge'],0.)
        self.assertEqual(summary['zero_charge_events'],2)
        self.assertEqual(summary['visibility_count_events'],{'0':2,'1':0,'2':0,'3':0})
        a=copy.deepcopy(zero);a.update(positive_charge=10.,charge_l1=.5,charge_l1_defined=True,absolute_charge_error=5.)
        b=copy.deepcopy(zero);b.update(positive_charge=90.,charge_l1=0.,charge_l1_defined=True,absolute_charge_error=0.)
        summary=_finding_coverage_summary([zero,a,b],3)
        self.assertAlmostEqual(summary['mean_active_event_charge_l1'],.25)
        self.assertAlmostEqual(summary['pooled_charge_l1'],.05)
        self.assertEqual(summary['fixed_cohort_events'],3)
        self.assertEqual(summary['positive_charge_events'],2)
        self.assertEqual(summary['nonzero_low_visible_events'],2)

    def write_cohort(self,folder,all_zero=False):
        obs,truth=folder/'observations.root',folder/'truth.root'
        meta=dict(source_dataset_uuid='coverage-fixture',source_group_partition_verified=True)
        records=[]
        count=2 if all_zero else 5
        with StageWriter(obs,'prepare',meta) as ow,StageWriter(truth,'truth',dict(meta,role='truth',tb_bin=4)) as tw:
            for entry in range(count):
                comp=np.zeros((3,12,64,512),np.float32)
                if not all_zero:
                    if entry==1:comp[0,0,0,100]=10.
                    elif entry>=2:
                        for alpha in range(entry-1):comp[alpha,:2,alpha,100+4*alpha]=100.
                q=comp.sum(0);key=dict(file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(100+entry))
                # Intentional poisoned extra fields: the evaluator's allowlist
                # must not pass these to observation-only sequential inference.
                ow.write(dict(key,positive_waveform=q,tb_ns=np.float64(40.),
                              MCTrack=np.array([987.]),si_alpha_mev=np.array([7.,8.,9.])),
                         dict(private_truth_diagnostic='must not reach inference'))
                tw.write(dict(key,truth_components=comp,truth_background=np.zeros_like(q)))
                records.append(dict(file_index=0,entry=entry,source_group=100+entry))
        split=folder/'split.json';split.write_text(json.dumps(dict(
            train=[dict(file_index=0,entry=900,source_group=900)],
            validation=[dict(file_index=0,entry=901,source_group=901)],test=records)))
        checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'not loaded by fixture inference')
        return dict(observations_root=str(obs),truth_root=str(truth),source_dataset_uuid='coverage-fixture'),split,checkpoint

    def test_root_fixed_cohort_strata_failures_and_sequential_observation_only_dispatch(self):
        import uproot
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);manifest,split,checkpoint=self.write_cohort(folder)
            observed_calls=[];standard_calls=[]
            def standard(q,cfg):
                standard_calls.append(dict(cfg))
                if float(q.sum())==600.:raise RuntimeError('fixture failed optimizer')
                return dict(fractions=fractions_for_observation(q,3),diagnostics=dict(
                    status='hypothesis_supported_NOT_physics_validated',ambiguous=False,seconds=.01))
            def sequential(record,cfg):
                observed_calls.append((set(record),dict(cfg)))
                self.assertEqual(set(record),{'positive_waveform','tb_ns'})
                q=record['positive_waveform']
                # Deliberately select a smaller observed hypothesis. It must be
                # padded only for matching, not called a failed 3-slot output.
                k=2
                return dict(fractions=fractions_for_observation(q,k),diagnostics=dict(
                    status='sequential_observation_hypothesis_supported_NOT_physics_validated',
                    ambiguous=False,selected_k=k,fixed_k=False,seconds=.02))
            output,report=folder/'evaluation.root',folder/'evaluation.json'
            with patch('reconstruction.algorithms.finding.find_event',side_effect=standard),patch(
                    'reconstruction.algorithms.sequential_finding.find_event',side_effect=sequential):
                result=evaluate_finding_comparison(manifest,split,checkpoint,output,report,
                    dict(methods=['hough','sequential_vertex'],tb_bin=4,finding=dict(frozen_setting=7)))
            self.assertEqual(len(observed_calls),5);self.assertEqual(len(standard_calls),5)
            self.assertTrue(all(cfg['method']=='sequential_vertex' and cfg['frozen_setting']==7 for _,cfg in observed_calls))
            self.assertTrue(all(cfg['method']=='hough' for cfg in standard_calls))
            h=result['summary']['hough'];seq=result['summary']['sequential_vertex']
            for summary in (h,seq):
                self.assertEqual(summary['N_events'],5)
                self.assertEqual(summary['zero_charge_events'],1)
                self.assertEqual(summary['positive_charge_events'],4)
                self.assertEqual(summary['nonzero_low_visible_events'],3)
                self.assertEqual(summary['visibility_count_events'],{'0':2,'1':1,'2':1,'3':1})
                self.assertEqual(sum(summary['exclusive_charge_visibility_strata'].values()),5)
            self.assertEqual(h['method_failed_events'],1)
            self.assertEqual(h['scored_active_events'],3)
            self.assertEqual(h['unscored_active_events'],1)
            self.assertAlmostEqual(h['pooled_charge_coverage'],610./1210.)
            self.assertEqual(h['mean_active_event_charge_l1'],0.)
            self.assertEqual(h['method_declaration_success_events'],3)
            self.assertAlmostEqual(h['method_declaration_success_fraction_all_cohort'],3./5.)
            self.assertEqual(seq['method_failed_events'],0)
            self.assertEqual(seq['scored_active_events'],4)
            self.assertGreater(seq['mean_active_event_charge_l1'],0.)
            rows=list(read_stage(output,allow_truth=True))
            self.assertEqual(len(rows),5)
            zero,_=rows[0]
            self.assertEqual(zero['positive_charge'],0.)
            self.assertTrue(zero['observed_zero_charge'])
            self.assertTrue(np.isnan(zero['charge_l1']).all())
            self.assertFalse(zero['method_declaration_success'].any())
            last,diagnostic=rows[-1]
            self.assertTrue(last['method_failed'][0]);self.assertFalse(last['prediction_valid'][0])
            self.assertTrue(np.isnan(last['predicted_fractions'][0]).all())
            self.assertIn('fixture failed optimizer',diagnostic['inference_diagnostics']['hough']['error_message'])
            self.assertTrue(np.isfinite(last['predicted_fractions'][1]).all())
            self.assertEqual(last['predicted_slot_count'][1],2)
            self.assertEqual(last['evaluation_padding_count'][1],1)
            self.assertEqual(last['recovered_slots'][1].sum(),2)
            self.assertEqual(diagnostic['inference_diagnostics']['sequential_vertex']['evaluation_slot_alignment']['zero_padded_alpha_slots'],1)
            with uproot.open(output) as f:
                self.assertEqual(f['performance/hough_positive_charge_defined_event_charge_l1'].values().sum(),3.)
                np.testing.assert_array_equal(f['performance/hough_all_cohort_reference_visible_count'].values(),[2,1,1,1])
                self.assertEqual(f['performance/sequential_vertex_all_cohort_reference_visible_count'].values().sum(),5.)
                summary_values=f['metrics'].arrays(library='np')
                scores=dict(zip(summary_values['name'],summary_values['value']))
                self.assertEqual(scores['hough_fixed_cohort_events'],5.)
                self.assertEqual(scores['hough_all_cohort_visible_0_events'],2.)
            # Strict JSON has null undefined scores, not fake zero or NaN tokens.
            saved=json.loads(report.read_text())
            self.assertIsNone(saved['per_event'][0]['metrics']['hough']['charge_l1'])

    def test_variable_k_padding_preserves_background_charge_and_missing_recovery(self):
        q=np.full((12,64,2),100.)
        truth=np.zeros((4,12,64,2));truth[0]=.3;truth[1]=.3;truth[2]=.4
        for selected_k in (1,2):
            raw=np.zeros((selected_k+1,12,64,2))
            raw[:selected_k]=.3
            raw[-1]=1.-.3*selected_k
            padded=_evaluation_fractions({'fractions':raw},3,q.shape)
            np.testing.assert_array_equal(padded[-1],raw[-1])
            np.testing.assert_array_equal(padded[:selected_k],raw[:-1])
            self.assertTrue((padded[selected_k:3]==0).all())
            np.testing.assert_allclose(padded.sum(0),raw.sum(0))
            metric=separation_metrics(padded,truth,q)
            self.assertEqual(int(metric['recovered'].sum()),selected_k)
            self.assertGreater(metric['charge_l1'],0.)
        too_many=np.zeros((5,12,64,2));too_many[-1]=1.
        with self.assertRaises(ValueError):_evaluation_fractions({'fractions':too_many},3,q.shape)

    def test_unexpected_more_than_maximum_slots_is_retained_as_explicit_failure(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);manifest,split,checkpoint=self.write_cohort(folder)
            def invalid_sequential(record,cfg):
                shape=(5,12,64,128)
                p=np.zeros(shape,np.float32);p[-1]=1.
                return dict(fractions=p,diagnostics=dict(ambiguous=False))
            output=folder/'evaluation.root'
            with patch('reconstruction.algorithms.sequential_finding.find_event',side_effect=invalid_sequential):
                result=evaluate_finding_comparison(manifest,split,checkpoint,output,folder/'evaluation.json',
                    dict(methods=['sequential_vertex']))
            summary=result['summary']['sequential_vertex']
            self.assertEqual(summary['fixed_cohort_events'],5)
            self.assertEqual(summary['method_failed_events'],5)
            self.assertEqual(summary['positive_charge_events'],4)
            self.assertEqual(summary['unscored_active_events'],4)
            self.assertIsNone(summary['pooled_charge_l1'])
            self.assertEqual(summary['all_cohort_positive_charge'],1210.)
            for row,diagnostic in read_stage(output,allow_truth=True):
                self.assertEqual(row['predicted_slot_count'][0],4)
                self.assertEqual(row['evaluation_padding_count'][0],0)
                self.assertFalse(row['prediction_valid'][0])
                self.assertIn('unsupported slot count',diagnostic['inference_diagnostics']['sequential_vertex']['error_message'])

    def test_component_centroid_ols_fixed_three_layer_peak_gate_and_absent_slots(self):
        comp=np.zeros((2,12,64,512),np.float32)
        for layer in range(4):comp[1,layer,28+layer,120]=100.
        found=dict(components=comp)
        theta=np.degrees(np.arctan(2./12.))
        result=_component_centroid_ols_angles(found,3,[1,0,2],True)
        self.assertAlmostEqual(result['theta_pad_deg'][0],theta,places=12)
        self.assertEqual(result['active_layers'].tolist(),[4,0,0])
        self.assertTrue(result['valid'][0]);self.assertTrue(np.isnan(result['theta_pad_deg'][1:]).all())
        # Two layers and a third at exactly 40 ADC cannot pass the fixed >40 gate.
        comp[1,2:]=0.;comp[1,2,30,120]=40.
        failed=_component_centroid_ols_angles(dict(components=comp),3,[1,0,2],True)
        self.assertEqual(failed['active_layers'][0],2)
        self.assertFalse(failed['valid'].any())
        self.assertTrue(np.isnan(failed['theta_pad_deg']).all())

    def test_saved_evaluation_contains_reference_angles_and_numeric_aligned_ols_without_truth_dependency(self):
        from reconstruction.stage_io import manifest as root_manifest
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);obs,truth=folder/'observed.root',folder/'truth.root'
            comp=np.zeros((3,12,64,512),np.float32)
            for layer in range(4):comp[0,layer,28+layer,120]=100.
            q=comp.sum(0);theta=np.degrees(np.arctan(2./12.))
            expected=np.array([theta,-10.,20.]);radians=np.radians(expected)
            directions=np.column_stack([np.sin(radians),np.cos(radians),np.zeros(3)])
            key=dict(file_index=np.int64(0),entry=np.int64(8),source_group=np.int64(88))
            meta=dict(source_dataset_uuid='angle-diagnostic-fixture',source_group_partition_verified=True)
            with StageWriter(obs,'prepare',meta) as writer:writer.write(dict(key,positive_waveform=q))
            with StageWriter(truth,'truth',dict(meta,role='truth',tb_bin=4)) as writer:
                writer.write(dict(key,truth_components=comp,truth_background=np.zeros_like(q),truth_direction_uvw=directions))
            split=folder/'split.json';split.write_text(json.dumps(dict(train=[],validation=[],
                test=[dict(file_index=0,entry=8,source_group=88)])))
            checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'not loaded')
            def sequential(record,cfg):
                coarse=record['positive_waveform'].reshape(12,64,128,4).sum(-1)
                p=np.zeros((3,12,64,128),np.float32);p[1,coarse>0]=1.;p[-1,coarse==0]=1.
                # Actual prediction order is reversed relative to truth; charge
                # alignment, not angle matching, must move this OLS track to slot0.
                return dict(fractions=p,components=np.stack([np.zeros_like(q),q]),
                    diagnostics=dict(status='sequential_observation_hypothesis_supported_NOT_physics_validated',ambiguous=False,seconds=0.))
            output=folder/'evaluation.root'
            with patch('reconstruction.algorithms.sequential_finding.find_event',side_effect=sequential):
                result=evaluate_finding_comparison(dict(observations_root=str(obs),truth_root=str(truth),
                    source_dataset_uuid=meta['source_dataset_uuid']),split,checkpoint,output,folder/'evaluation.json',
                    dict(methods=['sequential_vertex']))
            summary=result['summary']['sequential_vertex']
            self.assertEqual(summary['fixed_all_reference_slots'],3)
            self.assertEqual(summary['paired_initial_theta_error_slots'],1)
            self.assertAlmostEqual(summary['initial_theta_error_slot_coverage'],1./3.)
            self.assertAlmostEqual(summary['conditional_initial_minus_component_centroid_ols_theta_bias_deg'],0.,places=12)
            self.assertEqual(root_manifest(output)['role'],'evaluation')
            self.assertFalse(result['metadata']['angular_diagnostic']['truth_used_for_hit_selection'])
            self.assertFalse(result['metadata']['angular_diagnostic']['physical_3d'])
            # Standalone ROOT evaluation remains readable after original inputs
            # and reference/checkpoint files are unavailable.
            obs.unlink();truth.unlink();checkpoint.unlink()
            with self.assertRaises(ValueError):next(read_stage(output))
            row,diagnostic=next(read_stage(output,allow_truth=True))
            np.testing.assert_allclose(row['reference_initial_theta_pad_deg'],expected)
            np.testing.assert_array_equal(row['reference_pair_indices'],[[0,1],[0,2],[1,2]])
            np.testing.assert_allclose(row['reference_pair_delta_theta_pad_deg'],[theta+10.,20.-theta,30.])
            self.assertAlmostEqual(row['predicted_component_centroid_ols_theta_pad_deg'][0,0],theta,places=12)
            self.assertTrue(np.isnan(row['predicted_component_centroid_ols_theta_pad_deg'][0,1:]).all())
            self.assertAlmostEqual(row['initial_minus_component_centroid_ols_theta_error_deg'][0,0],0.,places=12)
            self.assertTrue(np.isnan(row['initial_minus_component_centroid_ols_theta_error_deg'][0,1:]).all())
            np.testing.assert_array_equal(row['component_centroid_ols_valid'],[[1,0,0]])
            self.assertEqual(row['pair_reference_visible'].sum(),0)
            self.assertEqual(row['pair_recovered'].sum(),0)

    def test_root_all_zero_fixed_cohort_has_no_l1_histogram_entries(self):
        import uproot
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);manifest,split,checkpoint=self.write_cohort(folder,all_zero=True)
            def standard(q,cfg):
                return dict(fractions=fractions_for_observation(q,3),diagnostics=dict(
                    status='hypothesis_supported_NOT_physics_validated',ambiguous=False,seconds=0.))
            output,report=folder/'evaluation.root',folder/'evaluation.json'
            with patch('reconstruction.algorithms.finding.find_event',side_effect=standard):
                result=evaluate_finding_comparison(manifest,split,checkpoint,output,report,dict(methods=['hough']))
            summary=result['summary']['hough']
            self.assertEqual(summary['fixed_cohort_events'],2)
            self.assertEqual(summary['visibility_count_events']['0'],2)
            self.assertIsNone(summary['pooled_charge_l1'])
            self.assertIsNone(summary['mean_active_event_charge_l1'])
            self.assertEqual(summary['method_declaration_success_events'],0)
            self.assertTrue(np.isfinite(summary['pooled_absolute_charge_error']))
            with uproot.open(output) as f:
                self.assertEqual(f['performance/hough_positive_charge_defined_event_charge_l1'].values().sum(),0.)
                self.assertEqual(f['performance/hough_all_cohort_reference_visible_count'].values().sum(),2.)
                values=f['metrics'].arrays(library='np');scores=dict(zip(values['name'],values['value']))
                self.assertTrue(np.isnan(scores['hough_pooled_charge_l1']))


if __name__=='__main__':unittest.main(verbosity=2)
