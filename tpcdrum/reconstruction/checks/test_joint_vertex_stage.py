import tempfile
from pathlib import Path
import unittest
import numpy as np

from reconstruction.algorithms.track_fit import _empty
from reconstruction.joint_vertex_stage import vertex_event
from reconstruction.stage_io import StageWriter,read_stage
from reconstruction.workflow import run_stage


def scene(dimension=2,k=3):
    record=_empty(k)
    vertex=np.array([2.,-40.,100.])[:dimension]
    v=np.arange(12)*12.
    slopes=np.array([[-.1,.02],[.1,-.04],[.2,.08]])[:k,:dimension-1]
    for a in range(k):
        p=np.c_[vertex[0]+slopes[a,0]*(v-vertex[1]),v]
        if dimension==3:p=np.c_[p,vertex[2]+slopes[a,1]*(v-vertex[1])]
        covariance=np.tile(np.eye(dimension)*.25,(12,1,1))
        record['fit_measurement_points_uv_mm'][a]=p[:,:2]
        record['fit_measurement_covariance_uv_mm2'][a]=covariance[:,:2,:2]
        record['fit_observed_layer_mask'][a]=1
        if dimension==3:
            record['fit_measurement_points_uvw_mm'][a]=p
            record['fit_measurement_covariance_mm2'][a]=covariance
            record['fit_points_uvw_mm'][a]=p
            d=np.array([slopes[a,0],1.,slopes[a,1]]);d/=np.linalg.norm(d)
            record['track_directions_uvw'][a]=d;record['track_origins_uvw_mm'][a]=p[0]
            record['fit_states'][a,:,:4]=np.c_[p[:,0],p[:,2],np.full(12,slopes[a,0]),np.full(12,slopes[a,1])]
            record['fit_valid_layer'][a]=1;record['fit_ok'][a]=1
    record.update(file_index=np.int64(0),entry=np.int64(0),source_group=np.int64(1),
        components=np.zeros((k,12,64,512),np.float32),energy_ok=np.ones(k,dtype=np.uint8),
        track_vertex_energy_mev=np.full(k,10.),material_paths_embedded_valid=np.int64(1))
    cfg={'drift':dict(calibrated_for_charge_mean=True,velocity_mm_per_ns=.01,t0_tb=0.,reference_w_mm=0.,sign=1.,tb_ns=40.),
         'vertex':{'method':'joint_tracks','joint':{'dimension':dimension,'vertex_v_scan_mm':[-500.,140.],
                   'process_noise_mode':'none','first_layer_half_extent_mm':6.}}}
    return record,{'fitting':{'physical_3d':dimension==3}},cfg,vertex


class JointVertexStageTests(unittest.TestCase):
    def test_projected_fit_two_tracks_adds_no_fake_precision_or_physical_z(self):
        row,diag,cfg,expected=scene(2,2)
        result,d=vertex_event(row,diag,cfg)
        self.assertEqual(d['vertex']['status'],'ok',d['vertex'])
        self.assertTrue(result['vertex_projected_valid']);self.assertFalse(result['vertex_valid'])
        np.testing.assert_allclose(result['vertex_projected_uv_mm'],expected,atol=1e-5)
        np.testing.assert_allclose(result['vertex_free_theta_uv_deg'],result['vertex_joint_theta_uv_deg'],atol=1e-7)
        self.assertTrue(np.isnan(result['vertex_joint_paths_uvw_mm'][:,:,2]).all())
        self.assertFalse(result['fit_ok'].any())

    def test_physical_joint_fit_invalidates_old_energy_and_keeps_original_measurements(self):
        row,diag,cfg,expected=scene(3,3)
        row.update(kinematics_method_accepted=np.int64(1),
                   kinematics_candidate_valid=np.int64(1),
                   kinematics_finding_assessment_available=np.int64(1),
                   kinematics_finding_ambiguous=np.int64(0))
        result,d=vertex_event(row,diag,cfg)
        self.assertEqual(d['vertex']['status'],'ok',d['vertex'])
        self.assertTrue(result['vertex_valid'])
        np.testing.assert_allclose(result['vertex_uvw_mm'],expected,atol=1e-4)
        self.assertFalse(result['energy_ok'].any())
        self.assertFalse(result['material_paths_embedded_valid'])
        self.assertFalse(result['kinematics_method_accepted'])
        self.assertFalse(result['kinematics_candidate_valid'])
        self.assertFalse(result['kinematics_finding_assessment_available'])
        self.assertEqual(result['kinematics_finding_ambiguous'],-1)
        self.assertTrue(np.isnan(result['fit_covariances']).all())
        self.assertTrue(result['vertex_direction_valid'].all())
        np.testing.assert_array_equal(result['pre_vertex_track_vertex_energy_mev'],[10.,10.,10.])
        self.assertFalse(result['vertex_joint_constraint_is_independent_measurement'])
        self.assertTrue(d['vertex_direction']['observation_only'])

    def test_root_handoff_and_failure_fields_are_stable(self):
        row,diag,cfg,_=scene(2,3)
        with tempfile.TemporaryDirectory() as folder:
            before=Path(folder)/'fit.root';after=Path(folder)/'vertex.root'
            with StageWriter(before,'fitting') as writer:
                writer.write(row,diag)
                bad={key:np.asarray(value).copy() for key,value in row.items()}
                bad['entry']=np.int64(1);bad['fit_observed_layer_mask'][:]=0
                writer.write(bad,diag)
            run_stage('vertex',before,after,cfg);before.unlink()
            rows=list(read_stage(after));self.assertEqual(len(rows),2)
            self.assertTrue(rows[0][0]['vertex_projected_valid'])
            self.assertFalse(rows[1][0]['vertex_projected_valid'])
            self.assertEqual(set(rows[0][0]),set(rows[1][0]))


if __name__=='__main__':unittest.main()
