"""Historical-MC timing math closure, not calibrated detector performance."""
import copy
import unittest

import numpy as np

from reconstruction.calibration.empirical_timing import (
    whole_layer_timing, unique_forward_crossing, fit_group_affine,
    bootstrap_group_affine, evaluate_frozen_map)
from reconstruction.preprocessing import PAD_U_MM


class EmpiricalTimingTests(unittest.TestCase):
    def crossing(self, points, v, **kwargs):
        config=dict(active_u_bounds_mm=[-36.95,36.95],max_segment_length_mm=2.)
        config.update(kwargs)
        return unique_forward_crossing(points,v,**config)

    def test_hand_calculated_whole_layer_mean_width_and_edges(self):
        q=np.zeros((12,64,512));q[0,0,100]=10;q[0,1,104]=30
        q[1,0,0]=50;q[1,0,511]=150
        before=q.copy()
        result=whole_layer_timing(q,minimum_layer_peak_adc=40.)
        self.assertAlmostEqual(result['mean_tb'][0],103.)
        self.assertAlmostEqual(result['width_tb'][0],np.sqrt(3.))
        self.assertAlmostEqual(result['mean_u_mm'][0],.25*PAD_U_MM[0]+.75*PAD_U_MM[1])
        self.assertFalse(result['valid_layer'][0])
        self.assertTrue(result['valid_layer'][1])
        self.assertAlmostEqual(result['mean_tb'][1],383.25)
        self.assertAlmostEqual(result['edge_fraction_low'][1],.25)
        self.assertAlmostEqual(result['edge_fraction_high'][1],.75)
        self.assertEqual(result['gate_counts']['total_layers'],12)
        self.assertEqual(result['gate_counts']['accepted_layers'],1)
        self.assertTrue(np.isnan(result['mean_tb'][2:]).all())
        self.assertFalse(result['calibrated_for_charge_mean'])
        np.testing.assert_array_equal(q,before)

    def test_subthreshold_pads_still_contribute_to_same_observed_statistic(self):
        q=np.zeros((12,64,512));q[0,0,10]=50;q[0,1,110]=25
        result=whole_layer_timing(q,minimum_layer_peak_adc=40.)
        self.assertTrue(result['valid_layer'][0])
        self.assertAlmostEqual(result['mean_tb'][0],(50*10+25*110)/75)
        scaled=whole_layer_timing(q*10)
        np.testing.assert_allclose(scaled['mean_tb'],result['mean_tb'],equal_nan=True)
        np.testing.assert_allclose(scaled['width_tb'],result['width_tb'],equal_nan=True)
        q[0,1,20]=-1
        with self.assertRaises(ValueError):whole_layer_timing(q)

    def test_interpolation_and_exact_midpoint_with_no_coordinate_transform(self):
        points=np.array([[0.,0.,100.],[.6,1.,101.],[1.2,2.,102.]])
        original=points.copy()
        result=self.crossing(points,.5)
        self.assertTrue(result['valid'])
        np.testing.assert_allclose(result['point_uvw_mm'],[.3,.5,100.5])
        self.assertEqual(result['segment_indices'],(0,1))
        self.assertFalse(result['coordinate_transform_applied'])
        exact=self.crossing(points,1.)
        self.assertTrue(exact['valid'])
        self.assertEqual(exact['crossing_count'],1)
        self.assertEqual(exact['segment_indices'],(1,1))
        self.assertEqual(exact['segment_length_mm'],0.)
        wrong=self.crossing(points,.5,coordinate_frame='geant4_xyz_mm')
        self.assertFalse(wrong['valid'])
        self.assertEqual(wrong['status'],'wrong_coordinate_frame')
        np.testing.assert_array_equal(points,original)

    def test_sloped_segment_gate_uses_full_length_not_just_layer_gap(self):
        points=np.array([[0.,0.,100.],[1.,1.,102.]])
        result=self.crossing(points,.5,max_segment_length_mm=2.)
        self.assertFalse(result['valid'])
        self.assertIn('crossing_segment_gap',result['failure_reasons'])
        self.assertAlmostEqual(result['segment_length_mm'],np.sqrt(6.))
        self.assertTrue(np.isnan(result['point_uvw_mm']).all())
        np.testing.assert_allclose(result['candidate_point_uvw_mm'],[.5,.5,101.])
        self.assertTrue(self.crossing(points,.5,max_segment_length_mm=3.)['valid'])

    def test_duplicate_reverse_multiple_and_coplanar_are_rejected(self):
        duplicate=[[0,0,0],[0,0,0],[0,1,1]]
        result=self.crossing(duplicate,.5)
        self.assertFalse(result['valid'])
        self.assertIn('duplicate_step_points',result['failure_reasons'])
        repeated=[[0,0,0],[0,2,2],[0,0,4],[0,2,6]]
        result=self.crossing(repeated,1.,max_segment_length_mm=4.)
        self.assertEqual(result['crossing_count'],3)
        self.assertIn('reverse_step',result['failure_reasons'])
        self.assertIn('multiple_crossings',result['failure_reasons'])
        coplanar=self.crossing([[0,0,0],[0,1,1],[.1,1,2],[0,2,3]],1.)
        self.assertIn('segment_in_target_plane',coplanar['failure_reasons'])
        self.assertFalse(coplanar['valid'])

    def test_outside_active_and_no_extrapolation_keep_failed_candidate(self):
        points=[[10,0,100],[10,1,101]]
        outside=self.crossing(points,.5,active_u_bounds_mm=[-1,1])
        self.assertFalse(outside['valid'])
        self.assertIn('outside_active_u',outside['failure_reasons'])
        self.assertEqual(outside['candidate_point_uvw_mm'][0],10.)
        no_crossing=self.crossing(points,-1.)
        self.assertIn('no_crossing_no_extrapolation',no_crossing['failure_reasons'])
        self.assertFalse(no_crossing['extrapolated'])
        self.assertTrue(np.isnan(no_crossing['candidate_point_uvw_mm']).all())
        boundary=self.crossing([[1,0,0],[1,1,0]],.5,active_u_bounds_mm=[-1,1])
        self.assertTrue(boundary['valid'])

    def test_equal_group_total_weight_prevents_sibling_count_dominance(self):
        x=np.r_[0.,np.ones(100),2.]
        y=np.r_[0.,np.full(100,2.),0.]
        groups=np.r_[0,np.full(100,1),2]
        fit=fit_group_affine(x,y,groups)
        self.assertTrue(fit['success'])
        np.testing.assert_allclose(fit['coefficients'],[2./3.,0.],atol=1.e-14)
        self.assertEqual(fit['fit_group_count'],3)
        self.assertEqual(fit['fit_row_count'],102)
        self.assertFalse(fit['calibrated_for_charge_mean'])
        self.assertFalse(fit['eligible_for_production'])
        self.assertGreater(np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0][0],1.9)

    def test_centered_affine_solve_identifiability_and_explicit_gates(self):
        x=np.array([1000.,1001.,1002.,1003.]);y=-50.+.36*x
        groups=np.array([0,0,1,1])
        fit=fit_group_affine(x,y,groups)
        self.assertTrue(fit['success'])
        np.testing.assert_allclose(fit['coefficients'],[-50.,.36],atol=1.e-10)
        self.assertLess(fit['centered_design_condition'],1.00001)
        bad=fit_group_affine([10,10],[3,4],[0,1])
        self.assertFalse(bad['success'])
        self.assertEqual(bad['status'],'nonidentifiable_timing_range')
        one=fit_group_affine([10],[3],[0])
        self.assertFalse(one['success'])
        self.assertEqual(one['coverage']['total_rows'],1)
        one_group=fit_group_affine([10,11],[3,4],[0,0])
        self.assertTrue(one_group['success'])
        self.assertFalse(one_group['group_uncertainty_identifiable'])

    def test_group_bootstrap_preserves_common_errors_and_row_duplication(self):
        group_count=12
        x=np.tile([10.,20.,30.],group_count)
        groups=np.repeat(np.arange(group_count),3)
        offsets=np.repeat(np.linspace(-2.,2.,group_count),3)
        y=7.+.4*x+offsets
        original=[array.copy() for array in (x,y,groups)]
        result=bootstrap_group_affine(x,y,groups,n_resamples=300,seed=91)
        self.assertTrue(result['success'])
        self.assertEqual(result['accepted_resamples'],300)
        self.assertGreater(result['covariance'][0,0],.01)
        self.assertLess(result['covariance'][1,1],1.e-26)
        repeat=20
        duplicated=bootstrap_group_affine(np.repeat(x,repeat),np.repeat(y,repeat),
            np.repeat(groups,repeat),n_resamples=300,seed=91)
        np.testing.assert_allclose(duplicated['coefficient_samples'],result['coefficient_samples'],atol=1.e-11)
        np.testing.assert_allclose(duplicated['covariance'],result['covariance'],atol=1.e-12)
        for before,after in zip(original,(x,y,groups)):np.testing.assert_array_equal(before,after)

    def test_bootstrap_records_failed_rank_resamples_instead_of_redrawing(self):
        result=bootstrap_group_affine([0.,1.],[2.,5.],[0,1],n_resamples=200,seed=2)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'],'partial_rank_valid_bootstrap')
        self.assertGreater(result['rejected_resamples'],0)
        self.assertEqual(result['accepted_resamples']+result['rejected_resamples'],200)
        self.assertTrue(np.isnan(result['resample_coefficients'][~result['resample_valid']]).all())
        np.testing.assert_allclose(result['coefficient_samples'],np.tile([2.,3.],(result['accepted_resamples'],1)))
        single=bootstrap_group_affine([0.,1.],[2.,5.],[0,0],n_resamples=5,seed=2)
        self.assertFalse(single['success'])
        self.assertEqual(single['status'],'insufficient_source_groups')

    def test_frozen_heldout_map_does_not_learn_heldout_offset(self):
        train_x=np.array([0.,1.,2.,3.]);train_y=2.+3.*train_x
        model=fit_group_affine(train_x,train_y,[0,0,1,1])
        before=copy.deepcopy(model)
        heldout_x=np.arange(10.,16.)
        heldout_y=2.+3.*heldout_x+100.
        result=evaluate_frozen_map(model,heldout_x,heldout_y,np.repeat([10,11,12],2),n_bootstrap=100,seed=4)
        self.assertFalse(result['coefficients_refitted'])
        self.assertAlmostEqual(result['summary']['bias_mm'],100.)
        np.testing.assert_allclose(result['summary']['group_ci']['bias_mm'],[100.,100.],atol=1.e-11)
        self.assertEqual(result['accepted_rows_outside_training_range'],6)
        self.assertEqual(result['training_source_group_overlap_count'],0)
        self.assertTrue(result['disjoint_from_training_source_groups'])
        self.assertFalse(result['calibrated_for_charge_mean'])
        np.testing.assert_array_equal(model['coefficients'],before['coefficients'])
        np.testing.assert_array_equal(model['fit_source_groups'],before['fit_source_groups'])
        self.assertEqual(model['coverage'],before['coverage'])

    def test_coverage_counts_every_gate_and_bins_retain_flow_and_empty_bins(self):
        model=fit_group_affine([0.,1.],[0.,1.],[0,1])
        x=np.array([1.,np.nan,3.,4.,5.,6.]);y=np.array([2.,3.,np.nan,5.,6.,7.])
        groups=np.repeat([0,1,2],2)
        eligible=np.array([1,1,1,0,1,1],bool)
        crossing=np.array([1,0,1,1,0,1],bool)
        values=np.array([-1.,0.,.5,1.,np.nan,3.])
        result=evaluate_frozen_map(model,x,y,groups,eligible=eligible,
            gate_masks={'unique_target_crossing':crossing},
            bins={'layer':{'values':values,'edges':[0.,1.,2.]}},n_bootstrap=100,seed=12)
        coverage=result['coverage']
        self.assertEqual(coverage['total_rows'],6)
        self.assertEqual(coverage['accepted_rows'],2)
        self.assertEqual(coverage['rejected_rows'],4)
        self.assertEqual(coverage['gates']['unique_target_crossing']['fail_rows'],2)
        self.assertEqual(sum(gate['rejected_here'] for gate in coverage['sequential_gates']),4)
        self.assertEqual(result['summary']['rows'],2)
        self.assertEqual(result['summary']['source_groups'],2)
        bins=result['bins']['layer']
        self.assertEqual(bins['underflow_rows'],1)
        self.assertEqual(bins['overflow_rows'],1)
        self.assertEqual(bins['nonfinite_rows'],1)
        self.assertEqual(bins['accepted_underflow_rows'],1)
        self.assertEqual(bins['accepted_overflow_rows'],1)
        self.assertEqual(bins['records'][0]['rows'],0)
        self.assertTrue(np.isnan(bins['records'][0]['bias_mm']))

    def test_group_evaluation_ci_does_not_shrink_from_duplicate_sibling_rows(self):
        model=fit_group_affine([0.,1.],[0.,1.],[0,1])
        x=np.tile([0.,1.,2.],10);groups=np.repeat(np.arange(10),3)
        y=x+np.repeat(np.linspace(-2.,2.,10),3)
        result=evaluate_frozen_map(model,x,y,groups,n_bootstrap=200,seed=31)
        repeat=evaluate_frozen_map(model,np.repeat(x,5),np.repeat(y,5),np.repeat(groups,5),n_bootstrap=200,seed=31)
        for name in ['bias_mm','rmse_mm','std_mm']:
            self.assertAlmostEqual(result['summary'][name],repeat['summary'][name])
            np.testing.assert_allclose(result['summary']['group_ci'][name],repeat['summary']['group_ci'][name],atol=1.e-12)

    def test_evaluation_labels_training_group_overlap_and_numeric_failure(self):
        model=fit_group_affine([0.,1.,2.,3.],[0.,1.,2.,3.],[0,0,1,1])
        result=evaluate_frozen_map(model,[0.,1.],[0.,1.],[0,3])
        self.assertEqual(result['training_source_group_overlap_count'],1)
        self.assertFalse(result['disjoint_from_training_source_groups'])
        overflow=dict(model,coefficients=np.array([0.,1.e308]))
        result=evaluate_frozen_map(overflow,[1.e308],[0.],[4])
        self.assertEqual(result['coverage']['accepted_rows'],0)
        self.assertEqual(result['coverage']['gates']['finite_frozen_prediction']['fail_rows'],1)
        self.assertTrue(np.isnan(result['summary']['bias_mm']))

    def test_invalid_configuration_is_not_a_silent_gate_or_calibration(self):
        with self.assertRaises(ValueError):self.crossing([[0,0,0],[0,1,1]],.5,max_segment_length_mm=0)
        with self.assertRaises(ValueError):whole_layer_timing(np.zeros((12,64,512)),minimum_layer_peak_adc=-1.)
        with self.assertRaises(ValueError):bootstrap_group_affine([0,1],[0,1],[0,1],n_resamples=1,seed=0)
        with self.assertRaises(ValueError):fit_group_affine([0,1],[0,1],[0,1],eligible=[True,np.nan])
        with self.assertRaises(ValueError):evaluate_frozen_map({'success':False},[0],[0],[0])
        model=fit_group_affine([0,1],[0,1],[0,1])
        with self.assertRaises(ValueError):evaluate_frozen_map(model,[0],[0],[0],bins={'bad':{'values':[0],'edges':[1,0]}})


if __name__=='__main__':unittest.main(verbosity=2)
