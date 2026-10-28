"""Synthetic software checks; they establish no detector/physics resolution."""
import copy
import inspect
import unittest
from unittest.mock import patch
import numpy as np

from reconstruction.algorithms import sequential_finding as seq
from reconstruction.algorithms import finding as wf
from reconstruction.checks.test_finding import synthetic
from reconstruction.preprocessing import space_points


def configuration(template,**sequential):
    return dict(k=3,activity_threshold_adc=1.,geometry=dict(scales_mm=[2,6,2],hough_directions=64,
        hough_pair_directions=32,proposal_points=400,ransac_trials=128,duplicate_angle_deg=2.,duplicate_distance=.4),
        waveform=dict(nonlinear_max_nfev=8,fit_pad_sigma=False,pad_sigma_mm=1.2,noise_sigma_adc=2.,
            shot_variance_scale=0.,fit_constant_background=False,max_hypotheses=4,
            pulse_template=dict(offset_tb=template[0].tolist(),values=template[1].tolist())),
        sequential=dict(maximum_first_seeds=2,maximum_pair_hypotheses=2,maximum_seconds_per_first=1,
            maximum_third_hypotheses_per_pair=2,vertex=dict(vertex_v_scan_mm=[-150,30],scan_points=33),**sequential))


def region(u=0.,status='resolved',opened=False,dimension=2):
    centers=np.array([[u],[u],[u]],float)
    cov=np.full((3,1,1),.01)
    if dimension==3:
        centers=np.c_[centers,np.full(3,10.)]
        cov=np.repeat((np.eye(2)*.01)[None],3,axis=0)
    return dict(status=status,dimension=dimension,point=np.array([u,-60.]) if status=='resolved' else None,
        v_interval_mm=np.array([-61.,-59.]),open_v_sides=[opened,False],
        transverse_region_envelope=dict(v_samples_mm=np.array([-61.,-60.,-59.]),centers_mm=centers,covariances_mm2=cov))


class SequentialFindingMathTests(unittest.TestCase):
    def test_materialized_nonnegative_background_survives_float_roundoff(self):
        rng=np.random.default_rng(78)
        forward=rng.uniform(0,4000,(3,12,64,512))
        q=forward.sum(0)
        fitted={'forward_components':forward,'forward_background':np.zeros_like(q)}
        with patch.object(wf,'_amplitude_fit',return_value=fitted):
            result,_=seq._materialize(q,{'tb_bin':4,'feature_time_scale':.36},np.zeros((3,4)),None,{})
        self.assertGreaterEqual(float(result['background'].min()),0.)
        self.assertGreaterEqual(float(result['components'].min()),0.)
        np.testing.assert_allclose(result['components'].sum(0)+result['background'],q,rtol=2e-6,atol=2e-6)

    def test_soft_peel_preserves_every_shared_positive_voxel_and_heldout(self):
        q=np.array([1.,10.,4.,0.]);p=np.array([1.,30.,2.,1.]);confidence=np.array([1.,1.,0.,1.])
        remaining,removed=seq.soft_peel(q,p,confidence,.8)
        np.testing.assert_allclose(remaining+removed,q)
        self.assertTrue((remaining[q>0]>0).all());self.assertEqual(remaining[2],q[2])
        np.testing.assert_array_equal(q,[1.,10.,4.,0.])
        with self.assertRaises(ValueError):seq.soft_peel(q,p,confidence,1.)
    def test_configuration_does_not_force_three_or_allow_untested_three(self):
        _,options=seq._configuration({'k':3})
        self.assertEqual(options['k_hypotheses'],[1,2,3])
        for cfg in ({'k':2,'sequential':{'k_hypotheses':[2,3]}},
                    {'k':3,'sequential':{'k_hypotheses':[3]}},{'k':True}):
            with self.assertRaises(ValueError):seq._configuration(cfg)
    def test_same_track_pulse_fragments_are_not_independent_seeds(self):
        a=np.array([2.,.1,420.,.02])
        self.assertTrue(seq._same_fragment(a,a+[.1,0.,1.,0.],{}))
        self.assertFalse(seq._same_fragment(a,a+[3.,0.,0.,0.],{}))
        self.assertFalse(seq._same_fragment(a,a+[0.,0.,12.,0.],{}))
    def test_region_changes_third_proposals_and_wrong_resolved_origin_rejected(self):
        _,_,template=synthetic([[0,0,420,0]])
        anchors=np.array([[12.,60.,420.,420.,1000.]])
        a=seq.region_conditioned_proposals(region(0.),anchors,[0.],None,template,{})
        b=seq.region_conditioned_proposals(region(8.),anchors,[0.],None,template,{})
        self.assertTrue(a and b);self.assertFalse(np.allclose(a[0],b[0]))
        self.assertTrue(seq.region_compatibility(a[1],region(0.),None,template,{})['eligible'])
        incompatible=seq.region_compatibility(np.array([30.,0.,420.,0.]),region(),None,template,{})
        self.assertFalse(incompatible['eligible'])
        extended=seq.region_compatibility(np.array([30.,0.,420.,0.]),region(status='extended_region',opened=True),None,template,{})
        self.assertIsNone(extended['eligible']);self.assertFalse(extended['within_scanned_envelope'])
    def test_raw_time_never_silently_becomes_physical_w(self):
        cal,status=seq._calibration({'feature_time_scale':.36},{})
        self.assertIsNone(cal);self.assertIn('projected_uv_only',status)
        cal,status=seq._calibration({'drift':{'calibrated_for_charge_mean':True}},{})
        self.assertIsNone(cal);self.assertIn('invalid_drift_calibration',status)
        _,_,template=synthetic([[0,0,420,0]])
        gate=seq.region_compatibility(np.array([0.,0.,420.,0.]),region(dimension=3),None,template,{})
        self.assertFalse(gate['available'])
    def test_profile_region_does_not_use_outside_confidence_depth_samples(self):
        r=region()
        r['transverse_region_envelope']=dict(v_samples_mm=np.array([-200.,-60.,50.]),
            centers_mm=np.array([[30.],[0.],[30.]]),covariances_mm2=np.full((3,1,1),.01))
        _,_,template=synthetic([[0,0,420,0]])
        gate=seq.region_compatibility(np.array([30.,0.,420.,0.]),r,None,template,{})
        self.assertFalse(gate['eligible'])
        outside=region(status='outside_reaction_volume')
        self.assertFalse(seq.region_compatibility(np.array([0.,0.,420.,0.]),outside,None,template,{})['eligible'])
        self.assertEqual(seq.region_conditioned_proposals(outside,np.array([[1.,0.,420.,420.,100.]]),[0.],None,template,{}),[])
    def test_correct_charge_mean_calibration_local_state_and_pad_offsets(self):
        p=np.array([[3.,.08,420.,.025]])
        q,_,template=synthetic(p)
        delays=np.tile(np.arange(12)[:,None]*.1,(1,64))
        cfg={'drift':dict(calibrated_for_charge_mean=True,velocity_mm_per_ns=.001,t0_tb=400.,reference_w_mm=2.,sign=1.,tb_ns=40.,pad_delay_tb=delays)}
        cal,status=seq._calibration(cfg,{'tb_ns':40.})
        self.assertIsNotNone(cal)
        origin,direction,cov,reference,ids=seq._local_state(q,cal,{})
        self.assertEqual(origin.shape,(3,));self.assertEqual(cov.shape,(4,4))
        self.assertGreater(np.linalg.eigvalsh(cov)[0],0.)
        self.assertAlmostEqual(direction[0],.08,delta=.012)
        self.assertAlmostEqual(direction[2],(.025-.1/12)*.04,delta=1e-4)
        np.testing.assert_allclose(seq._delay_at_uv(np.array([0.,2.]),np.array([0.,12.]),cal),[0.,.1])
    def test_space_point_proposals_preserve_observed_pad_identity_and_soft_residual(self):
        q,_,t=synthetic([[-4.,.03,420.,.01],[5.,-.02,425.,0.]])
        points,diagnostics=space_points(q,{'activity_threshold_adc':1.})
        record=dict(points,positive_waveform=q)
        cfg,_=seq._configuration(configuration(t));_,cfg=wf._event_input(q,cfg)
        full,weight=seq._proposal_points(q,record,cfg,'space_points')
        half,w_half=seq._proposal_points(q*.5,record,cfg,'space_points')
        np.testing.assert_allclose(full,half);np.testing.assert_allclose(w_half,weight*.5)
        self.assertTrue(np.isfinite(full).all());self.assertFalse(diagnostics['physical_3d'])
        with self.assertRaises(ValueError):seq._proposal_points(q,{'positive_waveform':q},cfg,'space_points')
    def test_numeric_history_schema_fixed_on_no_signal_and_local_failure(self):
        result=seq.find_event({'positive_waveform':np.zeros((12,64,512))},{'k':3})
        shapes={'sequential_history':(1,12),'sequential_seed_lines':(0,6),'sequential_seed_rank':(0,6),
            'sequential_confident_masks':(2,12,64,128),'sequential_seed_pair_indices':(0,2),
            'sequential_pair_regions':(0,10),'sequential_region_envelope':(0,7),
            'sequential_soft_residual':(12,64,512),'sequential_candidate_lines':(0,3,6),'sequential_candidate_scores':(0,8)}
        for name,shape in shapes.items():self.assertEqual(result[name].shape,shape,name)
        self.assertEqual(result['diagnostics']['selected_k'],0)
        self.assertTrue(result['diagnostics']['sequential_fallback'])
        self.assertEqual(list(inspect.signature(seq.find_event).parameters),['record','config','prior'])
    def test_external_prior_requires_heldout_blind_declaration_and_exact_time_shape(self):
        q,_,t=synthetic([[2.,0.,420.,0.]])
        prior=np.full((4,12,64,128),.25,dtype=np.float32)
        cfg=configuration(t)
        with self.assertRaisesRegex(ValueError,'heldout-pad blind'):seq.find_event({'positive_waveform':q},cfg,prior)
        cfg['waveform']['prior_is_heldout_blind']=True
        with self.assertRaisesRegex(ValueError,'prior must be'):seq.find_event({'positive_waveform':q},cfg,prior[...,:4])
    def test_local_sampling_covariance_not_precision_proportional_to_adc_charge(self):
        q,_,t=synthetic([[1.,.06,420.,.01]])
        a=seq._local_state(q,None,{})
        b=seq._local_state(q*100,None,{})
        np.testing.assert_allclose(a[2],b[2],rtol=1e-12,atol=1e-12)
        self.assertEqual(a[0].shape,(2,))


class SequentialFindingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generic unequal 3-body directions. No equal-energy/120-degree truth.
        cls.p=np.array([[-12.6,-.1,410.,.02],[6.3,.05,420.,-.01],[18.9,.15,430.,.03]])
        cls.q,_,cls.t=synthetic(cls.p,np.tile([1200.,800.,500.],(12,1)))
        cls.cfg=configuration(cls.t)
        cls.record={'positive_waveform':cls.q,'source_group':1001}
        cls.calls=[]
        original=wf._fit_hypothesis
        def spy(q,mask,*args,**kwargs):
            cls.calls.append((np.array(q,copy=True),np.array(mask,copy=True)))
            return original(q,mask,*args,**kwargs)
        with patch.object(wf,'_fit_hypothesis',side_effect=spy):
            cls.result=seq.find_event(cls.record,cls.cfg)
    def test_three_hypothesis_earns_heldout_support_and_vertex_really_proposes(self):
        d=self.result['diagnostics']
        self.assertEqual(d['sequential_status'],'completed');self.assertEqual(d['selected_k'],3)
        self.assertGreater(d['third_evidence_gain'],10.)
        self.assertTrue(any(x['proposal']=='vertex_region' for x in d['hypotheses']))
        self.assertLess(d['full_waveform_relative_l1'],1e-3)
        self.assertGreaterEqual(d['pair_hypothesis_count'],2)
        self.assertEqual({len(x['parameters']) for x in d['hypotheses']},{1,2,3})
    def test_every_joint_fit_uses_original_unpeeled_q_with_whole_neighbor_pad_holdout(self):
        mask=seq._train_mask(self.cfg)
        for observed,train in self.calls:
            np.testing.assert_array_equal(observed,self.q)
            np.testing.assert_array_equal(train,mask)
        remaining=self.result['sequential_soft_residual']
        np.testing.assert_allclose(remaining[~mask],self.q[~mask])
        self.assertTrue((remaining[self.q>0]>0).all())
        np.testing.assert_allclose(self.result['components'].sum(0)+self.result['background'],self.q,rtol=2e-6,atol=1e-4)
        np.testing.assert_allclose(self.result['fractions'].sum(0),1.,atol=2e-6)
    def test_pair_region_is_diagnostic_uv_and_not_an_independent_new_measurement(self):
        d=self.result['diagnostics']
        self.assertFalse(d['pair_vertex_physical_3d'])
        self.assertTrue(d['uncalibrated_feature_time_is_not_physical_w'])
        for r in d['pair_regions']:
            self.assertEqual(r['dimension'],2)
            self.assertFalse(r['pair_information_reused_as_independent_likelihood'])
            self.assertFalse(r['physical_uncertainty_validated'])
            self.assertFalse(r['quality']['distinct_fragment_ids_verified'])
    def test_clean_two_tracks_not_forced_into_three(self):
        q,_,t=synthetic(self.p[:2],np.tile([1200.,800.],(12,1)))
        r=seq.find_event({'positive_waveform':q},configuration(t))
        self.assertEqual(r['diagnostics']['selected_k'],2)
        self.assertFalse(r['diagnostics'].get('third_track_supported',False))
        self.assertLess(r['diagnostics']['full_waveform_relative_l1'],1e-3)
    def test_explicit_space_point_route_still_tests_original_neighbor_waveforms(self):
        q,_,t=synthetic(self.p[:2],np.tile([1200.,800.],(12,1)))
        measured,_=space_points(q,{'activity_threshold_adc':1.})
        r=seq.find_event(dict(measured,positive_waveform=q),configuration(t,proposal_mode='space_points'))
        self.assertEqual(r['diagnostics']['selected_k'],2)
        self.assertEqual(r['diagnostics']['proposal_mode'],'space_points')
        self.assertTrue(r['diagnostics']['final_fit_uses_original_waveform'])
        self.assertFalse(r['diagnostics']['pair_vertex_physical_3d'])
    def test_no_clean_seed_and_coincidence_falls_back_without_claiming_separation(self):
        q,_,t=synthetic([[2.,0.,420.,0.],[2.,0.,420.,0.]])
        cfg=configuration(t,minimum_clean_seed_layers=13)
        cfg['waveform']['nonlinear_max_nfev']=2
        r=seq.find_event({'positive_waveform':q},cfg)
        self.assertTrue(r['diagnostics']['sequential_fallback'])
        self.assertEqual(r['diagnostics']['sequential_status'],'no_clean_multilayer_seed')
        self.assertTrue(r['diagnostics']['ambiguous'])
        self.assertEqual(r['sequential_pair_regions'].shape,(0,10))
    def test_single_track_competes_even_after_spurious_pair_has_been_proposed(self):
        q,_,t=synthetic([[2.,.02,420.,0.]])
        cfg=configuration(t,k_hypotheses=[1,2]);cfg['k']=2
        configured,options=seq._configuration(cfg);_,configured=wf._event_input(q,configured)
        train=q*seq._train_mask(configured)[...,None]
        a=seq._seed_quality(np.array([2.,.02,420.,0.]),train,t,configured,options)
        b=seq._seed_quality(np.array([7.,.02,430.,0.]),train,t,configured,options)
        a['eligible']=True;b['eligible']=True;b['supported'][:]=True
        with patch.object(seq,'_pool',side_effect=[[a],[b]]):
            result=seq.find_event({'positive_waveform':q},cfg)
        self.assertFalse(result['diagnostics']['sequential_fallback'])
        self.assertEqual(result['diagnostics']['selected_k'],1)
        self.assertEqual(result['diagnostics']['chosen_pair_id'],-1)
        self.assertIsNone(result['diagnostics']['third_region_gate'])
    def test_checkpoint_prior_fallback_does_not_supply_checkpoint_and_prior_twice(self):
        q,_,t=synthetic([[2.,0.,420.,0.]])
        cfg=configuration(t,minimum_clean_seed_layers=13)
        cfg['waveform']['nonlinear_max_nfev']=0;cfg['checkpoint']='synthetic_mock_not_a_model.pt'
        prediction=dict(fractions=np.full((4,12,64,128),.25),checkpoint_sha256='mock_hash')
        with patch('reconstruction.algorithms.waveform_models.predict_checkpoint',return_value=prediction):
            r=seq.find_event({'positive_waveform':q},cfg)
        self.assertTrue(r['diagnostics']['sequential_fallback'])
        self.assertEqual(r['diagnostics']['checkpoint_sha256'],'mock_hash')


if __name__=='__main__':unittest.main()
