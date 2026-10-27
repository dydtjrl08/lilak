"""Paired-stage software tests: no actual detector accuracy is implied."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from reconstruction.stage_io import StageWriter,read_stage,manifest,sha256
from reconstruction.evaluation import evaluate_stage


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


if __name__=='__main__':unittest.main()


class EvaluationPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
    def fixture(self,accepted=True,corrupt_reported=False,direct_reference_four=False):
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
