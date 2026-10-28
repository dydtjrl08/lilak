"""Software fixtures for fixed-cohort angular curves, not detector performance."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import numpy as np

from reconstruction.angular_benchmark import (event_result,truth_pairs,wilson,
    AngularAccumulator,benchmark,summarize_rows,plot_benchmark,_attribution,configuration)
from reconstruction.stage_io import StageWriter,read_stage,manifest,jsonable


def fixture(k=2,permutation=None,charge=100.):
    truth=np.zeros((k,12,64,512),np.float32)
    theta=np.arange(k,dtype=float)*2.
    for a in range(k):truth[a,:,25+5*a,400+5*a]=charge
    direction=np.c_[np.tan(np.deg2rad(theta)),np.ones(k),np.zeros(k)]
    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    ref=dict(truth_components=truth,truth_background=np.zeros_like(truth[0]),
             truth_direction_uvw=direction,truth_vertex_uvw_mm=np.zeros(3))
    order=np.arange(k) if permutation is None else np.asarray(permutation)
    record=dict(positive_waveform=truth.sum(0),components=truth[order].copy(),
        background_waveform=np.zeros_like(truth[0]),fit_theta_pad_2d_ols_deg=theta[order],
        track_directions_uvw=direction[order],fit_ok=np.ones(k,dtype=np.uint8))
    return record,dict(fitting=dict(physical_3d=False)),ref


def compact_one_method(values):
    common=('pair_indices','delta_theta_xy_deg','pair_visible','observed_strata')
    return {name:(value if name in common else np.asarray(value)[None]) for name,value in values.items()}


class AngularMathTests(unittest.TestCase):
    def test_one_global_permutation_and_no_physical_direction_from_feature_coordinates(self):
        record,diag,ref=fixture(permutation=[1,0])
        result,note=event_result(record,diag,ref,config={'purity_min':.7,'recall_min':.7})
        np.testing.assert_array_equal(result['matching'],[1,0])
        self.assertTrue(result['pair_recovered_before'][0])
        self.assertFalse(result['pair_merge_before'][0]);self.assertFalse(result['pair_fake_split_before'][0])
        np.testing.assert_allclose(result['theta_2d_error_before'],0.,atol=1e-12)
        self.assertTrue(np.isnan(result['theta_3d_accepted_error_before']).all())
        self.assertIn('not MC timing',note['observed_strata'])

    def test_after_fit_does_not_rematch_slots_to_hide_a_swap(self):
        record,diag,ref=fixture(permutation=[1,0])
        after,_,_=fixture(permutation=[0,1])
        result,_=event_result(record,diag,ref,after,diag,{'purity_min':.7,'recall_min':.7})
        self.assertTrue(result['pair_recovered_before'][0])
        self.assertFalse(result['pair_recovered_after'][0])
        self.assertGreater(np.abs(result['theta_2d_error_after']).sum(),1.)

    def test_conditional_mc_recovery_is_not_method_confidence_acceptance(self):
        for flag in (None,0,1):
            record,diag,ref=fixture()
            if flag is not None:record['finding_ambiguous']=np.int64(flag)
            values,_=event_result(record,diag,ref)
            self.assertTrue(values['pair_recovered_before'][0])
            self.assertEqual(bool(values['pair_accepted_recovered_before'][0]),flag==0)
            self.assertEqual(int(values['pair_method_ambiguous_before'][0]),-1 if flag is None else flag)
            curve=summarize_rows([compact_one_method(values)],['method'],{'angle_bins_deg':[0.,5.,180.]})['curves'][0]
            self.assertEqual(curve['pair_recovered_before_fraction'],1.)
            self.assertEqual(curve['pair_accepted_recovered_before_fraction'],float(flag==0))
            self.assertEqual(curve['pair_accepted_recovered_before_denominator'],1)
            self.assertEqual(curve['pair_method_confidence_assessed_before_fraction'],float(flag is not None))
            if flag is None:self.assertTrue(np.isnan(curve['pair_method_ambiguous_before_fraction']))
            else:self.assertEqual(curve['pair_method_ambiguous_before_fraction'],float(flag))

    def test_diagnostic_ambiguity_fallback_and_conflicting_declarations_are_conservative(self):
        record,diag,ref=fixture();diag['finding']={'status':'geometric_proximity_only','ambiguous':True}
        values,_=event_result(record,diag,ref)
        self.assertTrue(values['pair_recovered_before'][0]);self.assertFalse(values['pair_accepted_recovered_before'][0])
        record['finding_ambiguous']=np.int64(0)
        values,_=event_result(record,diag,ref)
        self.assertTrue(values['pair_method_ambiguous_before'][0]);self.assertFalse(values['pair_accepted_recovered_before'][0])

    def test_merge_and_fake_split_are_explicit_frozen_charge_criteria(self):
        record,diag,ref=fixture()
        merged=copy.deepcopy(record);merged['components']=record['positive_waveform'][None]
        result,_=event_result(merged,diag,ref,config={'purity_min':.7,'recall_min':.7})
        self.assertFalse(result['pair_recovered_before'][0]);self.assertTrue(result['pair_merge_before'][0])
        self.assertFalse(result['pair_fake_split_before'][0])
        split=copy.deepcopy(record)
        split['components']=np.stack([ref['truth_components'][0]*.5,ref['truth_components'][0]*.5,ref['truth_components'][1]])
        result,_=event_result(split,diag,ref,config={'purity_min':.7,'recall_min':.7,'split_recall_min':.2})
        self.assertFalse(result['pair_recovered_before'][0]);self.assertTrue(result['pair_fake_split_before'][0])
        loose,_=event_result(split,diag,ref,config={'purity_min':.5,'recall_min':.5})
        self.assertTrue(loose['pair_recovered_before'][0])
        self.assertTrue(loose['pair_fake_split_before'][0])  # Recovery and structural failures are not exclusive.

    def test_identical_perfect_soft_decomposition_is_not_falsely_called_a_split(self):
        record,diag,ref=fixture()
        ref['truth_components'][1]=ref['truth_components'][0]
        record['components']=ref['truth_components'].copy();record['positive_waveform']=record['components'].sum(0)
        result,note=event_result(record,diag,ref)
        self.assertTrue(result['pair_recovered_before'][0]);self.assertFalse(result['pair_fake_split_before'][0])
        self.assertIn('not independently',note['perfect_overlap_identifiability'])

    def test_no_tracks_and_zero_signal_are_failures_or_ineligible_not_dropped(self):
        record,diag,ref=fixture()
        record['components']=np.zeros((0,12,64,512),np.float32)
        record['background_waveform']=record['positive_waveform'].copy()
        values,_=event_result(record,diag,ref)
        self.assertFalse(values['pair_recovered_before'][0]);np.testing.assert_array_equal(values['matching'],[-1,-1])
        accumulator=AngularAccumulator(['method'],{'angle_bins_deg':[0.,5.,180.]})
        accumulator.add(compact_one_method(values));rows=accumulator.curves()
        self.assertEqual(rows[0]['all_truth_pairs'],1);self.assertEqual(rows[0]['eligible_visible_pairs'],1)
        self.assertEqual(rows[0]['pair_recovered_before_fraction'],0.)
        self.assertTrue(np.isnan(rows[1]['pair_recovered_before_fraction']))
        record,diag,ref=fixture(charge=0.)
        values,_=event_result(record,diag,ref)
        accumulator=AngularAccumulator(['method'],{'angle_bins_deg':[0.,5.,180.]});accumulator.add(compact_one_method(values))
        self.assertEqual(accumulator.curves()[0]['all_truth_pairs'],1)
        self.assertEqual(accumulator.curves()[0]['eligible_visible_pairs'],0)
        self.assertTrue(np.isnan(accumulator.curves()[0]['pair_recovered_before_fraction']))

    def test_helper_failure_is_undefined_attribution_not_a_background_prediction_or_ambiguity(self):
        record,diag,ref=fixture()
        record.update(components=np.zeros((0,12,64,512),np.float32),
            background_waveform=record['positive_waveform'].copy())
        # K=0 without a failure flag remains a scored no-track prediction.
        score=_attribution(record,ref,configuration())
        self.assertEqual(score['charge_l1'],2.);self.assertFalse(score['finding_failed'])
        for flag in ({'finding_prediction_available':0},{'finding_failed':1}):
            failed=dict(record,**flag,finding_ambiguous=np.int64(1),
                vertex_free_theta_uv_deg=np.array([0.,2.]),vertex_joint_theta_uv_deg=np.array([0.,2.]),
                vertex_joint_comparison_valid=np.ones(2,np.uint8),
                vertex_projected_uv_mm=np.zeros(2),vertex_projected_valid=np.int64(1),
                vertex_free_projected_uv_mm=np.zeros(2),vertex_free_projected_valid=np.int64(1))
            score=_attribution(failed,ref,configuration())
            self.assertTrue(np.isnan(score['charge_l1']));self.assertTrue(np.isnan(score['precision']).all())
            self.assertTrue(np.isnan(score['recall']).all());self.assertFalse(score['good'].any())
            self.assertFalse(score['structural_metrics_known'])
            values,_=event_result(failed,diag,ref)
            self.assertTrue(values['pair_visible'][0]);self.assertFalse(values['pair_recovered_before'][0])
            self.assertTrue(values['pair_method_failed_before'][0]);self.assertFalse(values['pair_prediction_available_before'][0])
            self.assertEqual(values['pair_method_ambiguous_before'][0],-1)
            self.assertFalse(values['pair_method_confidence_assessed_before'][0])
            self.assertTrue(np.isnan(values['theta_2d_error_before']).all())
            self.assertTrue(np.isnan(values['vertex_2d_error_before']).all())
            curve=summarize_rows([compact_one_method(values)],['method'],{'angle_bins_deg':[0.,5.,180.]})['curves'][0]
            self.assertEqual(curve['eligible_visible_pairs'],1)
            self.assertEqual(curve['pair_recovered_before_denominator'],1)
            self.assertEqual(curve['pair_recovered_before_fraction'],0.)
            self.assertEqual(curve['pair_method_failed_before_fraction'],1.)
            self.assertEqual(curve['pair_prediction_available_before_fraction'],0.)
            self.assertEqual(curve['theta_2d_before_coverage'],0.)

    def test_failed_compact_after_stage_retains_before_permutation_and_fixed_pair_denominator(self):
        record,diag,ref=fixture(permutation=[1,0])
        after=dict(matched_prediction_index=np.array([-1,-1]),soft_precision=np.full(2,np.nan),
            soft_recall=np.full(2,np.nan),charge_l1=np.float64(np.nan),finding_failed=np.int64(1),
            finding_prediction_available=np.int64(0),finding_ambiguous=np.int64(1),
            theta_uv_ols_true_minus_fitted_deg=np.zeros(2),vertex_projected_uv_mm=np.zeros(2),
            vertex_projected_valid=np.int64(1))
        values,_=event_result(record,diag,ref,after,diag)
        np.testing.assert_array_equal(values['matching'],[1,0])
        self.assertTrue(values['pair_recovered_before'][0]);self.assertFalse(values['pair_recovered_after'][0])
        self.assertTrue(np.isnan(values['pair_purity_after']).all())
        self.assertTrue(np.isnan(values['theta_2d_error_after']).all())
        self.assertTrue(np.isnan(values['vertex_2d_error_after']).all())
        curve=summarize_rows([compact_one_method(values)],['method'],{'angle_bins_deg':[0.,5.,180.]})['curves'][0]
        self.assertEqual(curve['pair_recovered_after_denominator'],1)
        self.assertEqual(curve['pair_method_failed_after_count'],1)
        self.assertEqual(curve['theta_2d_after_coverage'],0.)

    def test_all_pairs_and_minimum_pair_are_declared(self):
        record,diag,ref=fixture(k=3)
        pairs,delta,_=truth_pairs(ref,{'pair_mode':'all_pairs'})
        self.assertEqual(pairs.shape,(3,2));np.testing.assert_allclose(sorted(delta),[2.,2.,4.])
        pairs,delta,_=truth_pairs(ref,{'pair_mode':'minimum_pair'})
        self.assertEqual(pairs.shape,(1,2));self.assertAlmostEqual(float(delta[0]),2.)

    def test_wilson_empty_zero_and_full_bins(self):
        self.assertTrue(all(np.isnan(v) for v in wilson(0,0)))
        p,low,high=wilson(0,10);self.assertEqual(p,0.);self.assertEqual(low,0.);self.assertAlmostEqual(high,.2775328,places=6)
        p,low,high=wilson(10,10);self.assertEqual(p,1.);self.assertAlmostEqual(low,.7224672,places=6);self.assertEqual(high,1.)
        with self.assertRaises(ValueError):wilson(2,1)

    def test_afterfit_coverage_uses_same_denominator_and_common_valid_subset_is_separate(self):
        accumulator=AngularAccumulator(['method'],{'angle_bins_deg':[0.,5.,180.]})
        for error,joint,valid in [(1.,.5,1),(10.,.1,0)]:
            record,diag,ref=fixture()
            theta=np.degrees(np.arctan2(ref['truth_direction_uvw'][:,0],ref['truth_direction_uvw'][:,1]))
            record.update(vertex_free_theta_uv_deg=theta+error,vertex_joint_theta_uv_deg=theta+joint,
                          vertex_joint_comparison_valid=np.full(2,valid,np.uint8))
            values,note=event_result(record,diag,ref)
            self.assertTrue(note['same_model_projected_theta_comparison'])
            accumulator.add(compact_one_method(values))
        curve=accumulator.curves()[0]
        self.assertEqual(curve['eligible_visible_pairs'],2)
        self.assertEqual(curve['theta_2d_before_n'],4);self.assertEqual(curve['theta_2d_after_n'],2)
        self.assertEqual(curve['theta_2d_before_coverage'],1.);self.assertEqual(curve['theta_2d_after_coverage'],.5)
        self.assertTrue(curve['theta_2d_after_coverage_decreased'])
        self.assertAlmostEqual(curve['theta_2d_common_before_rmse'],1.)
        self.assertAlmostEqual(curve['theta_2d_common_after_rmse'],.5)
        self.assertGreater(curve['theta_2d_before_rmse'],7.)

    def test_observed_strata_are_frozen_baseline_not_method_specific_selection(self):
        record,diag,ref=fixture();values,_=event_result(record,diag,ref)
        self.assertAlmostEqual(float(values['observed_strata'][0,0]),5.)
        self.assertEqual(values['observed_strata'][0,1],0.)
        self.assertEqual(values['observed_strata'][0,2],12.)
        self.assertEqual(values['observed_strata'][0,3],.5)
        c={'angle_bins_deg':[0.,5.,180.],'strata_bins':{'observed_drift_gap_tb':[0.,10.,100.]}}
        acc=AngularAccumulator(['method'],c);acc.add(compact_one_method(values))
        curves=acc.curves()
        self.assertEqual(curves[0]['eligible_visible_pairs'],1)
        selected=[r for r in curves if r['stratum_group_index']==1 and r['angle_bin']==0][0]
        self.assertEqual(selected['eligible_visible_pairs'],1)

    def test_numeric_result_rows_can_be_summarized_after_json_round_trip(self):
        record,diag,ref=fixture();values,_=event_result(record,diag,ref)
        row=json.loads(json.dumps(jsonable(compact_one_method(values))))
        summary=summarize_rows([row],['method'],{'angle_bins_deg':[0.,5.,180.]})
        self.assertEqual(summary['events'],1)
        self.assertEqual(summary['curves'][0]['pair_recovered_before_fraction'],1.)

    def test_absent_after_task_is_distinct_from_attempted_failed_after_fit(self):
        record,diag,ref=fixture();values,_=event_result(record,diag,ref)
        summary=summarize_rows([compact_one_method(values)],['method'],{'angle_bins_deg':[0.,5.,180.]})
        curve=summary['curves'][0]
        self.assertEqual(curve['phase_available_after_fraction'],0.)
        self.assertTrue(np.isnan(curve['theta_2d_after_coverage']))
        after=copy.deepcopy(record);after['fit_theta_pad_2d_ols_deg'][:]=np.nan
        values,_=event_result(record,diag,ref,after,diag)
        curve=summarize_rows([compact_one_method(values)],['method'],{'angle_bins_deg':[0.,5.,180.]})['curves'][0]
        self.assertEqual(curve['phase_available_after_fraction'],1.)
        self.assertEqual(curve['theta_2d_after_coverage'],0.)


class AngularRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
    def files(self,drop_second=False):
        meta=dict(source_dataset_uuid='angular-fixture',source_group_partition_verified=True)
        truth=self.folder/'truth.root';before=self.folder/'before.root';after=self.folder/'after.root'
        with StageWriter(truth,'truth',dict(meta,role='truth')) as writer:
            for entry in (2,1):
                r,d,t=fixture()
                writer.write(dict(t,file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(100+entry)))
        for path,joint in [(before,False),(after,True)]:
            with StageWriter(path,'jointfit' if joint else 'fitting',meta) as writer:
                for entry in ((1,) if drop_second and path==after else (1,2)):
                    r,d,t=fixture(permutation=[1,0]);r['fit_theta_pad_2d_ols_deg']+=1.
                    if joint:
                        theta=np.degrees(np.arctan2(t['truth_direction_uvw'][::-1,0],t['truth_direction_uvw'][::-1,1]))
                        r.update(vertex_free_theta_uv_deg=theta+1.,vertex_joint_theta_uv_deg=theta+.25,
                            vertex_joint_comparison_valid=np.ones(2,np.uint8),vertex_projected_uv_mm=np.array([.3,.4]),
                            vertex_projected_valid=np.int64(1),vertex_free_projected_uv_mm=np.array([.6,.8]),
                            vertex_free_projected_valid=np.int64(1))
                    writer.write(dict(r,file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(100+entry)),d)
        return truth,before,after
    def test_self_contained_root_curves_and_identity_order_after_upstream_deletion(self):
        truth,before,after=self.files();out=self.folder/'benchmark.root';jsonout=self.folder/'benchmark.json'
        result=benchmark({'algorithm':{'before':before,'after':after}},truth,out,jsonout,
            {'angle_bins_deg':[0.,5.,180.],'pair_mode':'minimum_pair','purity_min':.7,'recall_min':.7})
        self.assertEqual(result['cohort_events'],2);self.assertEqual(manifest(out)['role'],'evaluation')
        rows=list(read_stage(out,allow_truth=True));self.assertEqual([int(r['entry']) for r,_ in rows],[1,2])
        self.assertEqual(rows[0][0]['pair_recovered_before'].shape,(1,1))
        with self.assertRaises(ValueError):list(read_stage(out))
        for path in (truth,before,after):path.unlink()
        summary=summarize_rows(rows,['algorithm'],{'angle_bins_deg':[0.,5.,180.]})
        self.assertEqual(summary['events'],2);self.assertEqual(summary['curves'][0]['pair_recovered_before_fraction'],1.)
        self.assertAlmostEqual(summary['curves'][0]['theta_2d_after_rmse'],.25)
        self.assertAlmostEqual(summary['curves'][0]['vertex_2d_before_rmse'],1.)
        self.assertAlmostEqual(summary['curves'][0]['vertex_2d_after_rmse'],.5)
        import uproot
        with uproot.open(out) as source:
            self.assertEqual(source['angular_curves'].classname,'TTree')
            self.assertEqual(source['angular_curves'].num_entries,2)
        self.assertEqual(json.loads(jsonout.read_text())['curves'][1]['pair_recovered_before_fraction'],None)

    def test_missing_after_event_cannot_silently_reduce_cohort(self):
        truth,before,after=self.files(drop_second=True)
        output=self.folder/'bad.root'
        with self.assertRaisesRegex(ValueError,'lacks fixed cohort'):
            benchmark({'algorithm':{'before':before,'after':after}},truth,output)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
