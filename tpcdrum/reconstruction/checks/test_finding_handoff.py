"""Independent observed finding ROOT handoff; synthetic software closure only."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.finding_handoff import FindingHandoff, _optional_arrays
from reconstruction.preprocessing import PAD_U_MM
from reconstruction.stage_io import StageWriter, read_stage, manifest
from reconstruction.workflow import run_stage


def components(k):
    out=np.zeros((k,12,64,512),np.float32)
    for slot,slope in enumerate([.1,-.1,0.][:k]):
        for layer in range(4):
            u=slope*(12.*layer+12.)
            upper=int(np.searchsorted(PAD_U_MM,u));lower=upper-1
            f=(u-PAD_U_MM[lower])/(PAD_U_MM[upper]-PAD_U_MM[lower])
            out[slot,layer,lower,120+slot*4]=100.*(1.-f)
            out[slot,layer,upper,120+slot*4]=100.*f
    return out


def finding(comp,tb=4,optional=False):
    q=comp.sum(0);k=len(comp);coarse=q.reshape(12,64,512//tb,tb).sum(-1)
    assigned=np.concatenate([comp,np.zeros_like(q)[None]]).reshape(k+1,12,64,512//tb,tb).sum(-1)
    p=np.divide(assigned,coarse[None],out=np.zeros_like(assigned),where=coarse[None]>0);p[-1,coarse==0]=1.
    r=dict(components=comp.copy(),background=np.zeros_like(q),fractions=p,
        feature_lines=np.full((k,6),np.nan),diagnostics=dict(
            status='synthetic_observation_fixture',ambiguous=True,found_count=0 if q.sum()==0 else k))
    if optional:
        r.update(sequential_history=np.zeros((2,12)),sequential_seed_lines=np.zeros((k,6)),
            sequential_seed_rank=np.zeros((k,6)),sequential_seed_pair_indices=np.array([[0,1]],np.int64),
            sequential_confident_masks=np.zeros((2,12,64,512//tb),np.float32),
            sequential_pair_regions=np.full((1,10),np.nan),sequential_region_envelope=np.empty((0,7)),
            sequential_candidate_lines=np.full((1,3,6),np.nan),sequential_candidate_scores=np.zeros((1,8)),
            sequential_soft_residual=q*.1,forward_components=comp.copy(),
            forward_background=np.zeros_like(q),residual_waveform=np.zeros_like(q))
    return r


def record(entry,q):
    return dict(file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(100+entry),
                positive_waveform=q.copy(),tb_ns=np.float64(40.))


class FindingHandoffTests(unittest.TestCase):
    def test_forward_availability_requires_boolean_declaration_or_absent_finite_derivation(self):
        comp=components(1);q=comp.sum(0);fresh=finding(comp,optional=True)
        cases=[('absent',None,True,3),(True,True,True,2),(np.bool_(True),True,True,2),
               (False,True,False,1),(np.bool_(False),True,False,1),
               (0,True,False,4),(1,True,False,4),(None,True,False,4),('true',True,False,4)]
        for declaration,present,expected,basis in cases:
            with self.subTest(declaration=declaration,present=present):
                result=copy.deepcopy(fresh)
                if present:result['diagnostics']['forward_waveform_model_available']=declaration
                saved=_optional_arrays(result,q,1,128)
                self.assertEqual(bool(saved['waveform_model_available']),expected)
                self.assertEqual(int(saved['waveform_model_availability_basis']),basis)
                np.testing.assert_array_equal(saved['forward_components'],fresh['forward_components'])
        for name in ('forward_components','forward_background','residual_waveform'):
            for mode in ('missing','nan'):
                with self.subTest(name=name,mode=mode):
                    result=copy.deepcopy(fresh)
                    result['diagnostics']['forward_waveform_model_available']=True
                    if mode=='missing':del result[name]
                    else:result[name].flat[0]=np.nan
                    saved=_optional_arrays(result,q,1,128)
                    self.assertFalse(saved['waveform_model_available'])
                    self.assertEqual(int(saved['waveform_model_availability_basis']),0)
        wrong_k=copy.deepcopy(fresh);wrong_k['forward_components']=np.repeat(comp,2,axis=0)
        self.assertFalse(_optional_arrays(wrong_k,q,1,128)['waveform_model_available'])

    def test_real_baseline_waveform_and_sequential_fallback_availability_round_trip(self):
        from reconstruction.algorithms import finding as wf
        from reconstruction.algorithms import sequential_finding as seq
        from reconstruction.checks.test_finding import synthetic
        q,_,template=synthetic([[2.,.04,420.,.02]])
        config=dict(k=1,tb_bin=4,activity_threshold_adc=1.,
            geometry=dict(scales_mm=[2,6,2],hough_directions=32,hough_pair_directions=16,
                          proposal_points=200,ransac_trials=16),
            waveform=dict(nonlinear_max_nfev=0,fit_pad_sigma=False,pad_sigma_mm=1.2,
                noise_sigma_adc=2.,shot_variance_scale=0.,fit_constant_background=False,max_hypotheses=1,
                pulse_template=dict(offset_tb=template[0].tolist(),values=template[1].tolist())))
        with tempfile.TemporaryDirectory() as name:
            paths={method:Path(name)/(method+'.root') for method in ('tls','waveform_hypotheses','sequential_vertex')}
            predictions={method:[] for method in paths}
            with FindingHandoff(paths,config=config) as sink:
                for entry,observed in enumerate((q,np.zeros_like(q))):
                    r=record(entry,observed)
                    for method in paths:
                        fresh=(seq.find_event({'positive_waveform':observed},config) if method=='sequential_vertex'
                               else wf.find_event(observed,dict(config,method=method)))
                        predictions[method].append(fresh)
                        sink.write(method,r,found=fresh)
            for method,path in paths.items():
                rows=list(read_stage(path));self.assertEqual(len(rows),2)
                for entry,(saved,diagnostic) in enumerate(rows):
                    expected=method!='tls' and entry==0
                    self.assertEqual(bool(saved['waveform_model_available']),expected,(method,entry))
                    self.assertEqual(int(saved['waveform_model_availability_basis']),2 if expected else 1)
                    np.testing.assert_array_equal(saved['components'],predictions[method][entry]['components'])
                    np.testing.assert_array_equal(saved['component_fractions'],predictions[method][entry]['fractions'])
                    self.assertEqual(diagnostic['finding']['waveform_model_availability']['available'],expected)
                    self.assertFalse(diagnostic['finding']['waveform_model_availability']['physics_validated'])
                self.assertIn('Not fit quality',manifest(path)['waveform_model_availability_contract'])
                self.assertIn('3',manifest(path)['waveform_model_availability_basis_codes'])
            self.assertTrue(predictions['tls'][0]['components'].sum()>0)
            self.assertTrue(np.isnan(predictions['tls'][0]['forward_components']).all())
            self.assertTrue(predictions['sequential_vertex'][0]['diagnostics']['sequential_fallback'])
            self.assertTrue(predictions['sequential_vertex'][1]['diagnostics']['sequential_fallback'])

    def test_real_sequential_materializer_without_declaration_uses_documented_derived_basis(self):
        from reconstruction.algorithms import sequential_finding as seq
        from reconstruction.checks.test_finding import synthetic
        parameters=np.array([[2.,.04,420.,.02]])
        q,_,template=synthetic(parameters)
        fresh,_=seq._materialize(q,dict(tb_bin=4,feature_time_scale=.36),parameters,template,
            dict(pad_sigma_mm=1.2,noise_sigma_adc=2.,shot_variance_scale=0.,fit_constant_background=False))
        fresh['diagnostics']=dict(status='synthetic_actual_sequential_materializer',ambiguous=True,found_count=1)
        self.assertNotIn('forward_waveform_model_available',fresh['diagnostics'])
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'materialized.root'
            with FindingHandoff(dict(sequential_vertex=path),config=dict(k=1,tb_bin=4)) as sink:
                sink.write('sequential_vertex',record(0,q),found=fresh)
            saved,diagnostic=next(read_stage(path))
            self.assertTrue(saved['waveform_model_available'])
            self.assertEqual(int(saved['waveform_model_availability_basis']),3)
            self.assertIn('declaration absent',diagnostic['finding']['waveform_model_availability']['basis'])
            self.assertFalse(diagnostic['finding']['waveform_model_availability']['physics_validated'])

    def test_variable_slots_zero_failure_and_independent_uv_fitting_vertex(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);obs=folder/'original_observations.root';truth=folder/'original_truth.root';checkpoint=folder/'model.pt'
            a,b=components(3),components(1);zero=np.zeros((3,12,64,512),np.float32)
            qs=[a.sum(0),b.sum(0),zero.sum(0),a.sum(0)]
            meta=dict(role='observations',source_dataset_uuid='handoff-fixture',source_group_partition_verified=True)
            with StageWriter(obs,'prepare',meta) as writer:
                for i,q in enumerate(qs):writer.write(record(i,q))
            with StageWriter(truth,'truth',dict(meta,role='truth')) as writer:
                writer.write(dict(file_index=np.int64(0),entry=np.int64(0),truth_direction_uvw=np.ones((3,3))))
            checkpoint.write_bytes(b'checkpoint is NOT loaded by handoff')
            paths=dict(hough=folder/'hough.root',sequential_vertex=folder/'sequential.root')
            original_a=a.copy()
            with FindingHandoff(paths,meta,config=dict(finding=dict(k=3,tb_bin=4)),
                    checkpoint_sha256='a'*64,split_manifest_sha256='b'*64) as sink:
                for i,(r,_) in enumerate(read_stage(obs)):
                    # Poisoned fields must not leave the observation allowlist.
                    r.update(MCTrack=np.array([999.]),MCTagTPCDrum=np.array([888.]),
                             truth_direction_uvw=np.ones((3,3)),slot_permutation=np.array([2,1,0]),si_alpha_mev=np.array([8.,9.,10.]))
                    for method in paths:
                        if i==3:sink.write(method,r,error=RuntimeError('synthetic failed call'))
                        else:sink.write(method,r,found=finding([a,b,zero][i],optional=i==0))
            np.testing.assert_array_equal(a,original_a)
            obs.unlink();truth.unlink();checkpoint.unlink()
            rows=list(read_stage(paths['sequential_vertex']))
            self.assertEqual([int(r['finding_output_slot_count']) for r,_ in rows],[3,1,3,0])
            self.assertEqual([r['components'].shape[0] for r,_ in rows],[3,1,3,0])
            for i,(r,diag) in enumerate(rows):
                np.testing.assert_array_equal(r['positive_waveform'],qs[i])
                np.testing.assert_allclose(r['components'].sum(0)+r['background_waveform'],r['positive_waveform'],atol=2.e-5)
                np.testing.assert_allclose(r['component_fractions'].sum(0),1.,atol=2.e-5)
                self.assertFalse(any(key.startswith(('truth_','reference_','mc_target_')) for key in r))
                self.assertNotIn('slot_permutation',r);self.assertNotIn('MCTrack',r);self.assertNotIn('MCTagTPCDrum',r)
                self.assertNotIn('si_alpha_mev',r);self.assertNotIn('point_fractions',r)
                self.assertTrue(diag['handoff_observed_only'])
            np.testing.assert_array_equal(rows[0][0]['components'],a)
            self.assertEqual(rows[0][0]['sequential_history'].shape,(2,12))
            self.assertEqual(rows[1][0]['sequential_history'].shape,(0,12))
            self.assertTrue(rows[0][0]['waveform_model_available']);self.assertFalse(rows[1][0]['waveform_model_available'])
            zero_row=rows[2][0]
            self.assertTrue(zero_row['finding_observed_zero_charge']);self.assertEqual(zero_row['finding_found_count'],0)
            failed=rows[3][0]
            self.assertFalse(failed['finding_prediction_available']);self.assertTrue(failed['finding_failed'])
            self.assertTrue(failed['finding_ambiguous']);self.assertEqual(failed['feature_lines'].shape,(0,6))
            self.assertEqual(failed['component_fractions'].shape,(1,12,64,128))
            saved=manifest(paths['sequential_vertex'])
            self.assertEqual(saved['role'],'observations');self.assertEqual(saved['events'],4)
            self.assertEqual(saved['inference_config']['finding']['method'],'sequential_vertex')
            self.assertEqual(manifest(paths['hough'])['inference_config']['finding']['method'],'hough')
            self.assertNotIn('method',saved['comparison_config']['finding'])
            self.assertEqual(saved['checkpoint_sha256'],'a'*64);self.assertEqual(saved['split_manifest_sha256'],'b'*64)
            self.assertFalse(saved['physics_accuracy_evaluated']);self.assertFalse(saved['physical_w_calibration_claimed'])
            # No original ROOT, truth, checkpoint, or inference call is required.
            config=dict(drift={},fitting=dict(method='linear_rts',process_noise_mode='none',minimum_layers=3,
                minimum_layer_peak_adc=40.),vertex=dict(method='joint_tracks',joint=dict(dimension=2,
                process_noise_mode='none',minimum_layers=3,seed_layers=3,vertex_v_scan_mm=[-100.,20.],
                first_layer_half_extent_mm=6.,max_measurement_reduced_chi2=10.)))
            fitting_root,vertex_root=folder/'fitting.root',folder/'vertex.root'
            run_stage('fitting',paths['sequential_vertex'],fitting_root,config)
            paths['sequential_vertex'].unlink()
            run_stage('vertex',fitting_root,vertex_root,config)
            fitting_root.unlink()
            fitted=list(read_stage(vertex_root))
            self.assertEqual(len(fitted),4)
            self.assertTrue(fitted[0][0]['vertex_projected_valid'])
            np.testing.assert_allclose(fitted[0][0]['vertex_projected_uv_mm'],[0.,-12.],atol=1.e-6)
            self.assertFalse(fitted[0][0]['vertex_valid'])  # projected result is not a physical 3D vertex.
            for r,diag in fitted[1:]:
                self.assertFalse(r['vertex_projected_valid']);self.assertFalse(r['vertex_valid'])
            self.assertEqual(fitted[3][0]['fit_states'].shape,(0,12,5))

    def test_bad_closure_or_evaluation_fields_aborts_without_silent_event_drop(self):
        for failure in ('closure','evaluation'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as name:
                folder=Path(name);paths=dict(one=folder/'one.root',two=folder/'two.root')
                comp=components(1);r=record(0,comp.sum(0));fresh=finding(comp)
                with self.assertRaises(ValueError):
                    with FindingHandoff(paths,config=dict(k=3,tb_bin=4)) as sink:
                        sink.write('one',r,found=fresh)
                        broken=copy.deepcopy(fresh)
                        if failure=='closure':broken['components']*=.5
                        else:broken['diagnostics']['slot_permutation']=[0]
                        sink.write('two',r,found=broken)
                self.assertTrue(all(not path.exists() for path in paths.values()))

    def test_same_count_different_cohort_and_real_io_error_abort(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);paths=dict(one=folder/'one.root',two=folder/'two.root')
            comp=components(1);fresh=finding(comp)
            with self.assertRaises(ValueError):
                with FindingHandoff(paths) as sink:
                    sink.write('one',record(0,comp.sum(0)),found=fresh)
                    sink.write('two',record(1,comp.sum(0)),found=fresh)
            self.assertFalse(any(path.exists() for path in paths.values()))
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);paths=dict(one=folder/'one.root')
            with self.assertRaisesRegex(OSError,'synthetic disk error'):
                with FindingHandoff(paths) as sink:
                    with patch.object(sink.writers['one'],'write',side_effect=OSError('synthetic disk error')):
                        sink.write('one',record(0,comp.sum(0)),found=fresh)
            self.assertFalse(paths['one'].exists())

    def test_actual_sequential_zero_fallback_history_and_unknown_ambiguity(self):
        from reconstruction.algorithms.sequential_finding import find_event,HISTORY_COLUMNS
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'sequential.root'
            q=np.zeros((12,64,512),np.float32)
            # Actual algorithm execution on a synthetic zero event only. No user
            # data, checkpoint, live training or held-out inference is involved.
            found=find_event(record(0,q),dict(k=3,tb_bin=4,activity_threshold_adc=40.))
            self.assertEqual(found['sequential_history'].shape[1],len(HISTORY_COLUMNS))
            comp=components(1);unknown=finding(comp);unknown['diagnostics']['ambiguous']=0
            cfg=dict(finding=dict(k=3,tb_bin=4,method='original_global_method'),
                     drift=dict(reference_w_mm=0.,calibrated_for_charge_mean=False))
            before=copy.deepcopy(cfg)
            with FindingHandoff(dict(sequential_vertex=path),config=cfg) as sink:
                sink.write('sequential_vertex',record(0,q),found=found)
                sink.write('sequential_vertex',record(1,comp.sum(0)),found=unknown)
            rows=list(read_stage(path))
            np.testing.assert_array_equal(rows[0][0]['sequential_history'],found['sequential_history'])
            self.assertEqual(rows[1][0]['sequential_history'].shape,(0,len(HISTORY_COLUMNS)))
            self.assertTrue(rows[1][0]['finding_ambiguous'])
            self.assertFalse(rows[1][0]['finding_confidence_assessed'])
            self.assertTrue(rows[1][1]['finding']['ambiguous'])
            self.assertEqual(rows[1][1]['finding']['method_reported_ambiguous'],0)
            metadata=manifest(path)
            self.assertEqual(metadata['inference_config']['finding']['method'],'sequential_vertex')
            self.assertEqual(metadata['comparison_config']['finding']['method'],'original_global_method')
            self.assertEqual(cfg,before)

    def test_observed_si_module_identity_and_energies_survive_independent_reload(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'si_handoff.root'
            comp=components(1);r=record(0,comp.sum(0))
            r.update(si_edep_mev=np.array([5.5,.25]),has_observed_si_edep=np.int64(1),
                     si_center_uvw_mm=np.array([[10.,200.,100.],[-10.,200.,100.]]))
            source=dict(si_detector_names=['SiDet_RU','SiDet_U'],
                        truth_direction_uvw=[[1.,2.,3.]],slot_permutation=[2,1,0],private_truth='never copy')
            with FindingHandoff(dict(hough=path)) as sink:
                sink.write('hough',r,found=finding(comp),observation_diagnostic=source)
            del r,source
            saved,diagnostic=next(read_stage(path))
            np.testing.assert_array_equal(saved['si_edep_mev'],[5.5,.25])
            self.assertTrue(saved['has_observed_si_edep'])
            self.assertEqual(diagnostic['si_detector_names'],['SiDet_RU','SiDet_U'])
            self.assertEqual(set(diagnostic),{'si_detector_names','finding','handoff_observed_only'})
            self.assertNotIn('truth_direction_uvw',diagnostic);self.assertNotIn('slot_permutation',diagnostic)
            for index,names in enumerate((['SiDet_RU'],['SiDet_RU','SiDet_RU'])):
                badpath=folder/('bad_si_'+str(index)+'.root')
                bad=record(index,comp.sum(0));bad.update(si_edep_mev=np.array([5.5,.25]),has_observed_si_edep=np.int64(1))
                with self.assertRaises(ValueError):
                    with FindingHandoff(dict(hough=badpath)) as sink:
                        sink.write('hough',bad,found=finding(comp),observation_diagnostic=dict(observed_si_names=names))
                self.assertFalse(badpath.exists())

    def test_no_overwrite_wrong_metadata_and_fractional_point_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);existing=folder/'existing.root';existing.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):FindingHandoff(dict(one=existing))
            self.assertEqual(existing.read_bytes(),b'preserve')
            with self.assertRaises(ValueError):FindingHandoff(dict(one=folder/'new.root'),dict(role='evaluation'))
            # A physical calibration's coordinate reference is not an evaluation label.
            FindingHandoff(dict(one=folder/'calibration_metadata.root'),
                config=dict(drift=dict(reference_w_mm=0.,calibrated_for_charge_mean=False)))
            with self.assertRaises(ValueError):FindingHandoff(dict(one=folder/'fractional_k.root'),config=dict(k=2.5))
            with self.assertRaises(ValueError):FindingHandoff(dict(one=folder/'same.root',two=folder/'same.root'))
            comp=components(1);r=record(0,comp.sum(0));r.update(points_uvw_mm=np.zeros((1,3)),
                point_covariance_mm2=np.eye(3)[None],point_charge=np.ones(1),point_tb=np.ones(1),
                point_layer=np.array([.5]),point_pad=np.array([1]))
            with self.assertRaises(ValueError):
                with FindingHandoff(dict(one=folder/'new.root')) as sink:sink.write('one',r,found=finding(comp))
            self.assertFalse((folder/'new.root').exists())


if __name__=='__main__':unittest.main(verbosity=2)
