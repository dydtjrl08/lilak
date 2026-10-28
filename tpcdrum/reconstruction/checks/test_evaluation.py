"""Paired-stage software tests: no actual detector accuracy is implied."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from reconstruction.stage_io import StageWriter,read_stage,manifest,sha256
from reconstruction.evaluation import evaluate_stage,_fractions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
    def fixture(self,namespace='fixture',reference_namespace=None,source=17,reference_source=None,physical=True,local_swaps=False):
        c=np.zeros((2,12,64,512),np.float32);c[0,:,30,420]=100.;c[1,:,35,424]=50.
        pred=c[::-1].copy()
        if local_swaps:pred[:,6:]=c[:,6:]
        q=c.sum(0);v=np.array([[.1,1.,.02],[-.2,1.,.1]])
        v/=np.linalg.norm(v,axis=1,keepdims=True)
        obs=self.d/'obs.root';truth=self.d/'truth.root'
        om=dict(source_dataset_uuid=namespace,source_group_partition_verified=True)
        tm=dict(role='truth',source_dataset_uuid=reference_namespace or namespace,source_group_partition_verified=True,tb_bin=4)
        key=dict(file_index=np.int64(0),entry=np.int64(123),source_group=np.int64(source))
        with StageWriter(obs,'finding',om) as w:
            w.write(dict(key,positive_waveform=q,components=pred,background_waveform=np.zeros_like(q),
                track_directions_uvw=v[::-1].copy(),fit_ok=np.ones(2,dtype=np.uint8),
                fit_theta_pad_2d_ols_deg=np.degrees(np.arctan2(v[::-1,0],v[::-1,1]))),dict(fitting=dict(physical_3d=physical)))
        with StageWriter(truth,'reference',tm) as w:
            w.write(dict(key,source_group=np.int64(source if reference_source is None else reference_source),
                truth_components=c,truth_background=np.zeros_like(q),truth_direction_uvw=v))
        return obs,truth
    def run_eval(self,obs,truth):
        return evaluate_stage(obs,truth,self.d/'eval.root',self.d/'eval.json')
    def test_permutation_fixed_for_whole_event_and_directions(self):
        obs,truth=self.fixture();before=sha256(obs),sha256(truth)
        result=self.run_eval(obs,truth)
        self.assertEqual(result['summary']['mean_event_charge_l1'],0.)
        self.assertEqual(result['summary']['direction_valid_slots'],2)
        rows=list(read_stage(self.d/'eval.root',allow_truth=True))
        a,d=rows[0]
        np.testing.assert_array_equal(a['matched_prediction_index'],[1,0])
        np.testing.assert_allclose(a['direction_opening_error_deg'],0.,atol=2e-6)
        self.assertTrue(d['matching_reused_for_every_metric'])
        self.assertEqual(before,(sha256(obs),sha256(truth)))
    def test_cannot_fix_each_layer_with_different_permutation(self):
        obs,truth=self.fixture(local_swaps=True)
        result=self.run_eval(obs,truth)
        self.assertGreater(result['summary']['mean_event_charge_l1'],0.)
    def test_evaluation_root_is_not_downstream_observation_input(self):
        obs,truth=self.fixture();self.run_eval(obs,truth)
        self.assertEqual(manifest(self.d/'eval.root')['role'],'evaluation')
        with self.assertRaises(ValueError):list(read_stage(self.d/'eval.root'))
    def test_truth_root_cannot_substitute_observation_stage(self):
        obs,truth=self.fixture()
        with self.assertRaisesRegex(ValueError,'observation-role'):self.run_eval(truth,obs)
    def test_namespace_mismatch_refused(self):
        obs,truth=self.fixture(reference_namespace='unrelated-simulation')
        with self.assertRaisesRegex(ValueError,'namespace mismatch'):self.run_eval(obs,truth)
        self.assertFalse((self.d/'eval.root').exists())
    def test_source_group_mismatch_refused_and_no_partial_root_published(self):
        obs,truth=self.fixture(reference_source=18)
        with self.assertRaisesRegex(ValueError,'source_group mismatch'):self.run_eval(obs,truth)
        self.assertFalse((self.d/'eval.root').exists())
    def test_finite_feature_direction_does_not_become_calibrated_physical_direction(self):
        obs,truth=self.fixture(physical=False);result=self.run_eval(obs,truth)
        self.assertEqual(result['summary']['direction_valid_slots'],0)
        self.assertEqual(result['summary']['projected_ols_valid_slots'],2)
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertTrue(np.isnan(a['direction_opening_error_deg']).all())
        self.assertEqual(d['direction_status'],'missing_calibrated_physical_fit')
    def test_requested_missing_event_is_not_silently_dropped(self):
        obs,truth=self.fixture()
        with self.assertRaisesRegex(ValueError,'lacks requested'):
            evaluate_stage(obs,truth,self.d/'missing.root',config=dict(event_keys=[[0,999]]))
        self.assertFalse((self.d/'missing.root').exists())
    def test_existing_json_does_not_publish_partial_new_result(self):
        obs,truth=self.fixture();(self.d/'eval.json').write_text('keep')
        with self.assertRaises(FileExistsError):self.run_eval(obs,truth)
        self.assertFalse((self.d/'eval.root').exists())
    def test_reference_rows_out_of_order_are_matched_by_identity_not_row_number(self):
        c=np.zeros((1,12,64,512),np.float32);c[0,:,30,420]=100.
        meta=dict(source_dataset_uuid='out-of-order',source_group_partition_verified=True)
        obs=self.d/'obs.root';truth=self.d/'truth.root'
        with StageWriter(obs,'finding',meta) as ow:
            for e in (101,102):
                components=c*(e-100);q=components.sum(0)
                ow.write(dict(file_index=np.int64(0),entry=np.int64(e),source_group=np.int64(e+1000),
                              positive_waveform=q,components=components,background_waveform=np.zeros_like(q)))
        with StageWriter(truth,'truth',dict(meta,role='truth')) as tw:
            for e in (102,101):
                components=c*(e-100);q=components.sum(0)
                tw.write(dict(file_index=np.int64(0),entry=np.int64(e),source_group=np.int64(e+1000),
                              truth_components=components,truth_background=np.zeros_like(q)))
        result=self.run_eval(obs,truth)
        self.assertEqual(result['summary']['events'],2)
        self.assertEqual(result['summary']['mean_event_charge_l1'],0.)
        self.assertEqual([r['entry'] for r in result['per_event']],[101,102])

    def test_valid_no_track_prediction_is_penalized_but_explicit_failure_is_unscored(self):
        target=np.zeros((2,12,64,512),np.float32)
        target[0,:,30,420]=100.;target[1,:,35,424]=50.
        q=target.sum(0)
        row=dict(positive_waveform=q,components=np.zeros((0,12,64,512),np.float32),background_waveform=q.copy())
        ref=dict(truth_components=target,truth_background=np.zeros_like(q))
        missing=_fractions(row,ref,4)
        self.assertEqual(missing['charge_l1'],2.)
        self.assertTrue(missing['charge_l1_defined']);self.assertFalse(missing['finding_failed'])
        np.testing.assert_array_equal(missing['matched_prediction_index'],[-1,-1])
        np.testing.assert_array_equal(missing['precision'],[0.,0.])
        for flag in ({'finding_prediction_available':0},{'finding_failed':1}):
            failed=_fractions(dict(row,**flag),ref,4)
            self.assertTrue(failed['finding_failed']);self.assertFalse(failed['charge_l1_defined'])
            self.assertTrue(np.isnan(failed['charge_l1']));self.assertTrue(np.isnan(failed['absolute_charge_error']))
            self.assertTrue(np.isnan(failed['precision']).all());self.assertTrue(np.isnan(failed['recall']).all())
            self.assertFalse(failed['recovered'].any())
            np.testing.assert_array_equal(failed['true_visible'],[True,True])
        zero=np.zeros_like(q)
        m=_fractions(dict(row,positive_waveform=zero,background_waveform=zero),
            dict(ref,truth_components=np.zeros_like(target)),4)
        self.assertFalse(m['has_signal']);self.assertFalse(m['finding_failed'])
        self.assertFalse(m['charge_l1_defined']);self.assertTrue(np.isnan(m['charge_l1']))

    def test_mixed_frozen_root_cohort_keeps_failure_coverage_and_standalone_results(self):
        # One ambiguous-but-valid prediction, one valid K=0 abstention, and
        # one failed helper placeholder share an exact predeclared cohort.
        # The truth file is deliberately larger and stored in another order.
        target=np.zeros((2,12,64,512),np.float32)
        target[0,:,30,420]=100.;target[1,:,35,424]=50.;q=target.sum(0)
        directions=np.array([[0.,1.,0.],[.1,1.,.02]])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        vertex=np.array([1.,2.,3.])
        meta=dict(source_dataset_uuid='mixed-frozen-cohort',source_group_partition_verified=True)
        obs=self.d/'mixed_obs.root';truth=self.d/'larger_truth.root';out=self.d/'mixed_eval.root'
        with StageWriter(truth,'truth',dict(meta,role='truth')) as writer:
            for entry in (99,30,10,20):
                writer.write(dict(file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(entry+1000),
                    truth_components=target,truth_background=np.zeros_like(q),truth_direction_uvw=directions,
                    truth_vertex_uvw_mm=vertex,truth_kinetic_mev=np.array([10.,11.])))
        with StageWriter(obs,'vertex',meta) as writer:
            for entry in (30,20,10,99):
                valid=entry in (10,99);failed=entry==30
                # Deliberately stale physical fields must not score a missing
                # track or failed ancestor as a valid direction/vertex/energy.
                writer.write(dict(file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(entry+1000),
                    positive_waveform=q,components=target[::-1].copy() if valid else np.empty((0,12,64,512),np.float32),
                    background_waveform=np.zeros_like(q) if valid else q,
                    finding_prediction_available=np.uint8(not failed),finding_failed=np.uint8(failed),
                    finding_ambiguous=np.uint8(entry==10),track_directions_uvw=directions[::-1],
                    fit_ok=np.ones(2,np.uint8),fit_theta_pad_2d_ols_deg=np.degrees(np.arctan2(directions[::-1,0],directions[::-1,1])),
                    vertex_valid=np.int64(1),vertex_dimension=np.int64(3),vertex_uvw_mm=vertex,
                    track_vertex_energy_mev=np.array([11.,10.]),energy_ok=np.ones(2,np.uint8)),
                    dict(fitting=dict(physical_3d=True),vertex=dict(candidate_vertex_mm=vertex.tolist())))
        result=evaluate_stage(obs,truth,out,self.d/'mixed_eval.json',config={'event_keys':[[0,10],[0,20],[0,30]]})
        summary=result['summary']
        self.assertEqual(summary['events'],3);self.assertEqual(summary['positive_charge_events'],3)
        self.assertEqual(summary['zero_charge_events'],0);self.assertEqual(summary['finding_failed_events'],1)
        self.assertEqual(summary['prediction_available_events'],2);self.assertEqual(summary['scored_active_events'],2)
        self.assertEqual(summary['unscored_active_events'],1)
        self.assertAlmostEqual(summary['active_score_event_coverage'],2/3)
        self.assertAlmostEqual(summary['pooled_charge_coverage'],2/3)
        self.assertEqual(summary['mean_event_charge_l1'],1.);self.assertEqual(summary['pooled_charge_l1'],1.)
        self.assertEqual(summary['all_cohort_positive_charge'],5400.)
        self.assertEqual(summary['pooled_scored_positive_charge'],3600.)
        self.assertEqual(summary['visible_reference_slots'],6);self.assertEqual(summary['recovered_visible_slots'],2)
        self.assertEqual(summary['direction_reference_slots'],6);self.assertEqual(summary['direction_valid_slots'],2)
        self.assertEqual(summary['vertex_accepted_events'],1);self.assertEqual(summary['energy_valid_slots'],2)
        obs.unlink();truth.unlink()
        rows={int(row['entry']):row for row,_ in read_stage(out,allow_truth=True)}
        self.assertEqual(set(rows),{10,20,30})
        self.assertEqual(float(rows[10]['charge_l1']),0.)
        self.assertEqual(float(rows[20]['charge_l1']),2.)
        self.assertTrue(np.isnan(rows[30]['charge_l1']))
        for entry in (20,30):
            row=rows[entry]
            np.testing.assert_array_equal(row['matched_prediction_index'],[-1,-1])
            np.testing.assert_array_equal(row['visible_slots'],[1,1])
            self.assertFalse(row['recovered_slots'].any());self.assertFalse(row['direction_valid'].any())
            self.assertTrue(np.isnan(row['direction_opening_error_deg']).all())
            self.assertTrue(np.isnan(row['theta_uv_ols_true_minus_fitted_deg']).all())
            self.assertFalse(row['vertex_error_valid']);self.assertTrue(np.isnan(row['vertex_error_norm_mm']))
            self.assertFalse(row['kinetic_energy_valid'].any());self.assertTrue(np.isnan(row['kinetic_energy_error_mev']).all())
        np.testing.assert_array_equal(rows[20]['soft_precision'],[0.,0.])
        self.assertTrue(np.isnan(rows[30]['soft_precision']).all())
        self.assertFalse(rows[20]['finding_failed']);self.assertTrue(rows[30]['finding_failed'])
        import uproot
        with uproot.open(out) as source:
            np.testing.assert_array_equal(source['performance/all_cohort_finding_failed'].values(),[2.,1.])


if __name__=='__main__':unittest.main()


class EvaluationPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
    def fixture(self,accepted=True,corrupt_reported=False,direct_reference_four=False,method_assessed=True):
        from reconstruction.evaluation import _independent_threebody
        mass=4.;ground=12.
        p=np.array([[2.,0.,.3],[-.6,1.2,-.1],[-1.4,-1.2,-.2]])
        rest=np.column_stack((p,np.sqrt(mass*mass+np.sum(p*p,axis=1))))
        beta=np.array([.18,-.05,.07]);b2=beta@beta;gamma=1/np.sqrt(1-b2)
        transform=np.eye(4);transform[:3,:3]+=(gamma-1)*np.outer(beta,beta)/b2
        transform[:3,3]=gamma*beta;transform[3,:3]=gamma*beta;transform[3,3]=gamma
        lab=rest@transform.T
        perm=np.array([2,0,1]);pred=lab[perm];computed=_independent_threebody(pred,ground)
        c=np.zeros((3,12,64,512),np.float32)
        for i in range(3):c[i,:,30+i*2,420+i]=100.+50*i
        q=c.sum(0);directions=lab[:,:3]/np.linalg.norm(lab[:,:3],axis=1,keepdims=True)
        key=dict(file_index=np.int64(0),entry=np.int64(11),source_group=np.int64(777))
        meta=dict(source_dataset_uuid='synthetic-generic-three-body',source_group_partition_verified=True)
        vertex=np.array([1.,2.,3.]);delta=np.array([.1,-.2,.3])
        arrays=dict(key,positive_waveform=q,components=c[perm],background_waveform=np.zeros_like(q),
            track_directions_uvw=directions[perm],fit_ok=np.ones(3,dtype=np.uint8),
            vertex_valid=np.int64(accepted),vertex_dimension=np.int64(3),vertex_uvw_mm=vertex+delta,
            track_vertex_energy_mev=pred[:,3]-mass,energy_ok=np.full(3,int(accepted),dtype=np.uint8),
            kinematics_ok=np.int64(accepted),kinematics_candidate_valid=np.int64(1),
            alpha_four_vectors_mev=pred,invariant_mass_mev=np.float64(computed['mass']+(0.02 if corrupt_reported else 0.)),
            excitation_mev=np.float64(computed['excitation']+(0.02 if corrupt_reported else 0.)),
            com_opening_angles_deg=computed['angles']+np.array([1.,0.,0.])*(1 if corrupt_reported else 0),
            com_pair_indices=np.array([[0,1],[0,2],[1,2]]),
            dalitz_xy=computed['dalitz']+(np.array([.01,-.02]) if corrupt_reported else 0.))
        if method_assessed:arrays['kinematics_method_accepted']=np.int64(accepted)
        with StageWriter(self.d/'obs.root','kinematics',meta) as w:
            w.write(arrays,dict(fitting=dict(physical_3d=True),vertex=dict(candidate_vertex_mm=(vertex+delta).tolist())))
        refs=dict(key,truth_components=c,truth_background=np.zeros_like(q),truth_vertex_uvw_mm=vertex,
            truth_kinetic_mev=lab[:,3]-mass,truth_momentum_uvw_mev=lab[:,:3],truth_direction_uvw=directions)
        if direct_reference_four:refs['truth_four_vectors_mev']=lab
        with StageWriter(self.d/'truth.root','truth',dict(meta,role='truth')) as w:w.write(refs)
        return lab,vertex,delta
    def run_eval(self,masses=True):
        return evaluate_stage(self.d/'obs.root',self.d/'truth.root',self.d/'eval.root',config=dict(masses=dict(alpha_mev=4.,carbon12_ground_mev=12.)) if masses else {})
    def test_exact_permuted_generic_threebody_closes_without_120_constraint(self):
        lab,vertex,delta=self.fixture();result=self.run_eval()
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertEqual(result['summary']['kinematics_accepted_events'],1)
        self.assertEqual(result['summary']['energy_valid_slots'],3)
        np.testing.assert_allclose(a['reference_four_vectors_mev'],lab,atol=1e-12)
        np.testing.assert_array_equal(a['matched_prediction_index'],[1,2,0])
        np.testing.assert_allclose(a['vertex_error_uvw_mm'],delta,atol=1e-12)
        np.testing.assert_allclose(a['kinetic_energy_error_mev'],0.,atol=1e-12)
        self.assertAlmostEqual(float(a['invariant_mass_error_mev']),0.,places=11)
        self.assertAlmostEqual(float(a['excitation_error_mev']),0.,places=11)
        np.testing.assert_allclose(a['com_pair_angle_error_deg'],0.,atol=1e-10)
        np.testing.assert_allclose(a['dalitz_error_xy'],0.,atol=1e-12)
        self.assertGreater(float(np.ptp(a['reference_com_pair_angles_deg'])),1.)
        self.assertTrue(d['physical_observable_evaluation']['no_120_degree_constraint'])
    def test_unaccepted_candidate_not_mixed_with_accepted_accuracy(self):
        self.fixture(accepted=False);r=self.run_eval()
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertEqual(r['summary']['kinematics_accepted_events'],0)
        self.assertEqual(r['summary']['kinematics_candidate_events'],1)
        self.assertTrue(np.isnan(a['invariant_mass_error_mev']))
        self.assertAlmostEqual(float(a['candidate_invariant_mass_error_mev']),0.,places=11)
        self.assertTrue(np.isnan(a['vertex_error_norm_mm']))
        self.assertTrue(np.isfinite(a['vertex_candidate_error_norm_mm']))
        self.assertTrue(np.isnan(a['kinetic_energy_error_mev']).all())
        np.testing.assert_allclose(a['kinetic_energy_candidate_error_mev'],0.,atol=1e-12)

    def test_legacy_ok_without_explicit_method_acceptance_remains_candidate_only(self):
        self.fixture(accepted=True,method_assessed=False);r=self.run_eval()
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertEqual(r['summary']['kinematics_accepted_events'],0)
        self.assertEqual(r['summary']['kinematics_candidate_events'],1)
        self.assertFalse(a['kinematics_accepted_error_valid'])
        self.assertTrue(np.isnan(a['invariant_mass_error_mev']))
        self.assertTrue(np.isnan(a['com_pair_angle_error_deg']).all())
        self.assertAlmostEqual(float(a['candidate_invariant_mass_error_mev']),0.,places=11)
        np.testing.assert_allclose(a['candidate_com_pair_angle_error_deg'],0.,atol=1e-10)
    def test_missing_masses_never_silently_use_an_assumed_alpha_mass(self):
        self.fixture();r=self.run_eval(masses=False)
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertEqual(r['summary']['reference_four_vector_events'],0)
        self.assertTrue(np.isnan(a['candidate_invariant_mass_error_mev']))
        self.assertEqual(r['summary']['energy_valid_slots'],3)
    def test_independent_recompute_exposes_wrong_reported_stage_observables(self):
        self.fixture(corrupt_reported=True);r=self.run_eval()
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertAlmostEqual(float(a['invariant_mass_error_mev']),.02,places=10)
        self.assertAlmostEqual(float(a['invariant_mass_stage_consistency_mev']),.02,places=10)
        self.assertAlmostEqual(float(np.max(abs(a['com_pair_angle_error_deg']))),1.,places=9)
        self.assertGreater(float(np.max(abs(a['dalitz_error_xy']))),.001)
        np.testing.assert_allclose(a['dalitz_stage_consistency_xy'],[.01,-.02],atol=1e-12)
    def test_explicit_initial_fourvectors_allow_mass_without_inventing_excitation_reference(self):
        self.fixture(direct_reference_four=True);r=self.run_eval(masses=False)
        a,d=next(read_stage(self.d/'eval.root',allow_truth=True))
        self.assertEqual(r['summary']['reference_four_vector_events'],1)
        self.assertAlmostEqual(float(a['invariant_mass_error_mev']),0.,places=11)
        self.assertTrue(np.isnan(a['reference_excitation_mev']))
        self.assertTrue(np.isnan(a['excitation_error_mev']))
