"""Software/math tests. Synthetic examples are NOT detector accuracy evidence."""
import inspect
import tempfile
from pathlib import Path
import unittest
import numpy as np

from reconstruction.algorithms.finding import (find_event,_basis,_amplitude_fit,_fit_hypothesis,
                                                _pulse_template,PAD_U_MM)
from reconstruction.algorithms import geometry_core
from reconstruction.training_finding import split_source_groups,separation_metrics


def synthetic(parameters,amplitudes=None):
    offsets=np.arange(-20.,21.)
    template=(offsets,np.exp(-.5*(offsets/3.)**2),'synthetic_known_template')
    a=np.asarray(parameters,dtype=float).reshape(-1,4)
    basis=_basis(a,template,1.2)
    if amplitudes is None:amplitudes=np.full((12,len(a)),600.)
    components=basis*np.asarray(amplitudes).T[:,:,None,None]
    return components.sum(0),components,template


class FindingMathTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[-2.,.04,420.,.008],[2.,-.025,422.,-.005]])
        self.q,self.true,self.template=synthetic(self.p,np.tile([900.,650.],(12,1)))
        self.mask=(np.arange(64)[None,:]+2*np.arange(12)[:,None])%4!=0
        self.options=dict(pad_sigma_mm=1.2,noise_sigma_adc=1.,shot_variance_scale=0.,fit_constant_background=False,nonlinear_max_nfev=0)
    def test_nnls_recovers_known_overlapping_templates_and_neighbors(self):
        fit=_amplitude_fit(self.p,self.q,self.mask,self.template,self.options,keep=True)
        np.testing.assert_allclose(fit['amplitudes'],np.tile([900.,650.],(12,1)),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(fit['prediction'],self.q,rtol=1e-8,atol=1e-8)
        self.assertLess(fit['chi2'],1e-15)
        self.assertTrue(np.isfinite(fit['conditional_amplitude_std']).all())
    def test_wrong_single_waveform_rejected_by_heldout_neighbors(self):
        correct=_fit_hypothesis(self.q,self.mask,self.p,self.template,self.options,optimize=False)
        wrong=_fit_hypothesis(self.q,self.mask,np.array([[0.,.0075,421.,0.]]),self.template,self.options,optimize=False)
        self.assertLess(correct['heldout_chi2'],1e-15)
        self.assertGreater(wrong['heldout_chi2'],1000.)
        self.assertGreater(wrong['selection_score'],correct['selection_score'])
    def test_heldout_pad_values_do_not_change_fitted_training_amplitudes(self):
        changed=self.q.copy();changed[~self.mask]*=2.
        a=_amplitude_fit(self.p,self.q,self.mask,self.template,self.options)
        b=_amplitude_fit(self.p,changed,self.mask,self.template,self.options)
        np.testing.assert_array_equal(a['amplitudes'],b['amplitudes'])
        a=_fit_hypothesis(self.q,self.mask,self.p,self.template,self.options,optimize=False)
        b=_fit_hypothesis(changed,self.mask,self.p,self.template,self.options,optimize=False)
        self.assertGreater(b['heldout_chi2'],a['heldout_chi2']+1000.)
    def test_exactly_coincident_pulses_are_rank_deficient(self):
        p=np.repeat(self.p[:1],2,axis=0)
        q,_,template=synthetic(p)
        fit=_amplitude_fit(p,q,self.mask,template,self.options)
        self.assertTrue(np.all(fit['template_pair_correlation']>.999999))
        self.assertTrue(np.isinf(fit['conditional_amplitude_std']).any())
    def test_native_wide_pads_integrate_more_uniform_centered_response(self):
        q,_,t=synthetic([[0.,0.,420.,0.]])
        # The two central wide pads at -1,+1 are symmetric; no asymmetric indexing.
        i=np.where(PAD_U_MM==-1.)[0][0];j=np.where(PAD_U_MM==1.)[0][0]
        np.testing.assert_allclose(q[:,i],q[:,j])
    def test_all_geometric_methods_conserve_charge_and_keep_background(self):
        cfg=dict(k=2,activity_threshold_adc=1.,geometry=dict(scales_mm=[2,6,2],hough_directions=32,hough_pair_directions=16,proposal_points=200,ransac_trials=64))
        for method in ('hough','ransac','tls'):
            r=find_event(self.q,dict(cfg,method=method))
            self.assertEqual(r['components'].shape,(2,12,64,512))
            self.assertTrue(np.all(r['components']>=0))
            np.testing.assert_allclose(r['components'].sum(0)+r['background'],self.q,rtol=2e-6,atol=1e-5)
            np.testing.assert_allclose(r['fractions'].sum(0),1.,atol=2e-6)
            self.assertFalse(r['diagnostics']['calibrated_physical_w'])
    def test_geometry_permutation_invariance_with_outliers(self):
        rng=np.random.default_rng(7);v=np.linspace(0,132,12)
        points=np.column_stack((.1*v,v,3.+.04*v));charge=np.ones(12)
        cfg=dict(scales_mm=[1,6,2],min_points=3,hough_directions=32,hough_pair_directions=16,ransac_trials=64)
        first=geometry_core.infer(points,charge,1,'ransac',seed=9,config=cfg)
        order=rng.permutation(12)
        second=geometry_core.infer(points[order],charge[order],1,'ransac',seed=9,config=cfg)
        np.testing.assert_allclose(first['lines'],second['lines'])
        np.testing.assert_allclose(first['fractions'][order],second['fractions'])
    def test_missing_signal_no_forced_three_lines(self):
        r=find_event(np.zeros((12,64,512)),dict(k=3))
        self.assertEqual(r['diagnostics']['status'],'no_active_signal')
        self.assertTrue(np.isnan(r['feature_lines']).all())
        self.assertEqual(float(r['components'].sum()),0.)
    def test_invalid_inputs_and_truth_free_interface(self):
        bad=self.q.copy();bad[0,0,0]=-1
        with self.assertRaises(ValueError):find_event(bad,{})
        with self.assertRaises(ValueError):find_event(self.q,dict(tb_bin=3))
        with self.assertRaises(ValueError):find_event(self.q,dict(method='cnn_hough'))
        self.assertEqual(list(inspect.signature(find_event).parameters),['positive_waveform','config','prior'])
    def test_source_siblings_never_cross_split(self):
        groups=['A:0','A:0','A:1','A:1','A:2','A:3','B:0','B:1']
        labels,mapping=split_source_groups(groups,42)
        self.assertEqual(labels[0],labels[1]);self.assertEqual(labels[2],labels[3])
        self.assertEqual(set(labels),{'train','validation','test'})
        self.assertEqual(labels,split_source_groups(groups,42)[0])
    def test_end_to_end_waveform_hypotheses_choose_two_on_separated_synthetic_tracks(self):
        q,_,template=synthetic([[-6.,.03,420.,.01],[6.,-.025,424.,-.01]],np.tile([900.,650.],(12,1)))
        cfg=dict(k=2,activity_threshold_adc=1.,geometry=dict(scales_mm=[2,6,2],hough_directions=64,
            hough_pair_directions=32,proposal_points=400,ransac_trials=128),waveform=dict(compare_k=True,
            k_candidates=[1,2],max_hypotheses=6,nonlinear_max_nfev=12,pad_sigma_mm=1.2,noise_sigma_adc=1.,
            shot_variance_scale=0.,fit_constant_background=False,
            pulse_template=dict(offset_tb=template[0].tolist(),values=template[1].tolist())))
        r=find_event(q,cfg)
        self.assertEqual(r['diagnostics']['selected_k'],2)
        self.assertEqual(set(x['k'] for x in r['diagnostics']['hypotheses']),{1,2})
        self.assertLess(r['diagnostics']['full_waveform_relative_l1'],1e-3)
        np.testing.assert_allclose(r['components'].sum(0)+r['background'],q,atol=1e-4,rtol=2e-6)
        self.assertTrue(r['diagnostics']['forward_waveform_model_available'])
        self.assertFalse(r['diagnostics']['physics_accuracy_evaluated'])
    def test_end_to_end_coincident_tracks_remain_ambiguous(self):
        q,_,t=synthetic([[2.,.01,420.,0.],[2.,.01,420.,0.]])
        cfg=dict(k=2,activity_threshold_adc=1.,geometry=dict(scales_mm=[2,6,2],hough_directions=32,
            hough_pair_directions=16,proposal_points=200,ransac_trials=64),waveform=dict(max_hypotheses=3,
            nonlinear_max_nfev=10,noise_sigma_adc=1.,shot_variance_scale=0.,fit_constant_background=False,
            pulse_template=dict(offset_tb=t[0].tolist(),values=t[1].tolist())))
        r=find_event(q,cfg)
        self.assertTrue(r['diagnostics']['ambiguous'])
        self.assertNotEqual(r['diagnostics']['status'],'hypothesis_supported_NOT_physics_validated')
    def test_signal_only_on_heldout_pad_returns_missing_evidence_status(self):
        q=np.zeros((12,64,512));q[0,0,420]=100.
        r=find_event(q,dict(k=3,activity_threshold_adc=1.))
        self.assertEqual(r['diagnostics']['status'],'no_train_pad_support_for_holdout')
        np.testing.assert_array_equal(r['background'],q)
    def test_full_event_external_prior_cannot_leak_into_holdout_evidence(self):
        prior=np.full((3,12,64,128),1./3,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'heldout-pad'):
            find_event(self.q,dict(k=2,activity_threshold_adc=1.),prior)
    def test_perfect_permuted_separation_scores_zero_error(self):
        charge=np.full((12,64,4),100.)
        truth=np.zeros((3,12,64,4));truth[0]=.7;truth[1]=.3
        metric=separation_metrics(truth[[1,0,2]],truth,charge)
        self.assertEqual(metric['charge_l1'],0.)
        self.assertEqual(metric['overlap_fraction_mae'],0.)
        self.assertTrue(metric['recovered'].all())
    def test_charge_l1_above_one_is_not_an_error_probability(self):
        charge=np.ones((12,64,4));truth=np.zeros((2,12,64,4));truth[0]=1.
        wrong=truth[::-1].copy()
        metric=separation_metrics(wrong,truth,charge)
        self.assertEqual(metric['charge_l1'],2.)


class FindingNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:raise unittest.SkipTest('PyTorch not installed; classical tests remain runnable')
        cls.torch=torch;torch.set_num_threads(1)
    def test_network_simplex_backward_and_pit_permutation(self):
        import torch
        from reconstruction.algorithms.waveform_models import NativeWaveformResUNet,features_numpy
        from reconstruction.training_finding import pit_charge_loss
        q,_,_=synthetic([[-4.,0.,420.,0.],[4.,0.,424.,0.]])
        x=torch.from_numpy(features_numpy(q,16)[None])
        model=NativeWaveformResUNet(2,16,width=4,dropout=0.)
        p=model(x)
        self.assertEqual(tuple(p.shape),(1,3,12,64,32))
        torch.testing.assert_close(p.sum(1),torch.ones_like(p[:,0]))
        target=torch.zeros_like(p);target[:,0]=.7;target[:,1]=.3
        charge=torch.ones_like(p[:,0])
        a=pit_charge_loss(p,target,charge)
        b=pit_charge_loss(p[:,[1,0,2]],target,charge)
        torch.testing.assert_close(a,b)
        a.backward();self.assertTrue(any(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in model.parameters()))
    def test_untrained_checkpoint_is_explicitly_refused(self):
        import torch
        from reconstruction.algorithms.waveform_models import NativeWaveformResUNet,load_finding_checkpoint
        model=NativeWaveformResUNet(1,64,width=4)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'untrained.pt'
            torch.save(dict(contract=model.contract(),model=model.state_dict(),training_completed=False),p)
            with self.assertRaisesRegex(ValueError,'untrained'):load_finding_checkpoint(p)
    def test_streaming_root_training_runs_and_records_disjoint_source_groups(self):
        from reconstruction.stage_io import StageWriter
        from reconstruction.training_finding import train_finding,RootFindingDataset
        from reconstruction.algorithms.waveform_models import load_finding_checkpoint
        import json
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);obs=folder/'observations.root';truth=folder/'truth.root'
            with StageWriter(obs,'prepared',dict(source_group_partition_verified=True)) as ow,StageWriter(truth,'truth',dict(role='truth')) as tw:
                for e in range(6):
                    q,comp,_=synthetic([[float(e%3)-1.,.01,420.+e,0.]])
                    identity=dict(file_index=np.int64(0),entry=np.int64(e),source_group=np.int64(e))
                    ow.write(dict(identity,positive_waveform=q.astype(np.float32)))
                    tw.write(dict(identity,truth_components=comp.astype(np.float32)))
            manifest=dict(observations_root=str(obs),truth_root=str(truth),source_dataset_uuid='SYNTHETIC_SOFTWARE_TEST')
            ds=RootFindingDataset(manifest,k=1,tb_bin=64)
            x,target,charge=ds[0];self.assertEqual(tuple(x.shape),(3,12,64,8));ds.close()
            result=train_finding(manifest,folder/'training',dict(k=1,tb_bin=64,width=4,epochs=1,batch_size=1,device='cpu',dropout=0.))
            model,checkpoint,_=load_finding_checkpoint(result['model_path'])
            self.assertTrue(checkpoint['training_completed'])
            self.assertEqual(result['provenance']['test_evaluation_count'],1)
            self.assertFalse(result['provenance']['physics_accuracy_evaluated'])
            splits=json.loads((folder/'training/split_manifest.json').read_text())
            sets={k:set(r['source_group'] for r in rows) for k,rows in splits.items()}
            self.assertFalse(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['test']&sets['validation'])
    def test_unverified_original_group_provenance_is_not_production_training(self):
        from reconstruction.stage_io import StageWriter
        from reconstruction.training_finding import train_finding
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);q,c,_=synthetic([[0.,0.,420.,0.]])
            with StageWriter(d/'obs.root','prepared') as ow,StageWriter(d/'truth.root','truth',dict(role='truth')) as tw:
                for e in range(3):
                    ids=dict(file_index=np.int64(0),entry=np.int64(e),source_group=np.int64(e))
                    ow.write(dict(ids,positive_waveform=q.astype(np.float32)))
                    tw.write(dict(ids,truth_components=c.astype(np.float32)))
            m=dict(observations_root=str(d/'obs.root'),truth_root=str(d/'truth.root'),source_dataset_uuid='unknown-original-mapping')
            with self.assertRaisesRegex(ValueError,'provenance is unverified'):
                train_finding(m,d/'refused',dict(k=1,tb_bin=64,width=4,epochs=1,device='cpu'))
    def test_evaluation_role_cannot_be_training_observations(self):
        from reconstruction.stage_io import StageWriter
        from reconstruction.training_finding import RootFindingDataset
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);q,c,_=synthetic([[0.,0.,420.,0.]])
            ids=dict(file_index=np.int64(0),entry=np.int64(0),source_group=np.int64(0))
            with StageWriter(d/'evaluation.root','comparison',dict(role='evaluation')) as ew:
                ew.write(dict(ids,positive_waveform=q.astype(np.float32)))
            with StageWriter(d/'truth.root','truth',dict(role='truth')) as tw:
                tw.write(dict(ids,truth_components=c.astype(np.float32)))
            m=dict(observations_root=str(d/'evaluation.root'),truth_root=str(d/'truth.root'),source_dataset_uuid='fixture')
            with self.assertRaisesRegex(ValueError,'observation-role'):
                RootFindingDataset(m,k=1,tb_bin=64)


if __name__=='__main__':unittest.main()
