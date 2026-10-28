import unittest
import numpy as np
from reconstruction.vertex_measurements import local_state,observed_tracks
from test_track_fit import waveforms,configuration
from reconstruction.algorithms.track_fit import fit_event


class LocalVertexObservationTests(unittest.TestCase):
    def test_original_uv_support_exists_without_fabricating_physical_z(self):
        cfg=configuration();cfg['drift']={}
        data=waveforms();updates,_=fit_event(data,cfg);data.update(updates)
        tracks,covs,layers,slots=observed_tracks(data,cfg,2)
        self.assertEqual(len(tracks),2)
        state=local_state(tracks[0],covs[0])
        self.assertAlmostEqual(state['direction'][0]/state['direction'][1],1/12)
        self.assertTrue(np.all(np.linalg.eigvalsh(state['covariance'])>0))
        with self.assertRaisesRegex(ValueError,'calibration'):observed_tracks(data,cfg,3)

    def test_local_state_uses_first_observed_layers_and_preserves_scaling(self):
        v=np.arange(12)*12.;p=np.c_[2.+.1*v,v,30.-.05*v]
        p[6:,0]+=20.  # End-of-track deviation must not rotate the seed tangent.
        c=np.tile(np.diag([.25,1.,.5]),(12,1,1))
        a=local_state(p,c,4);b=local_state(p,c*4,4)
        np.testing.assert_allclose(a['state'],[2.,30.,.1,-.05],atol=1e-10)
        np.testing.assert_allclose(b['covariance'],4*a['covariance'])
        np.testing.assert_array_equal(a['layers_used'],[0,1,2,3])

    def test_singular_or_invalid_covariance_and_duplicate_depth_rejected(self):
        p=np.c_[np.arange(4),np.arange(4)*12.];c=np.tile(np.eye(2),(4,1,1))
        with self.assertRaises(ValueError):local_state(p,np.zeros_like(c))
        p[1,1]=p[0,1]
        with self.assertRaises(ValueError):local_state(p,c)


if __name__=='__main__':unittest.main()
