import tempfile
from pathlib import Path
import unittest
import numpy as np

from reconstruction.stage_io import StageWriter,read_stage,manifest
from reconstruction.workflow import run_stage


class WorkflowHandoffChecks(unittest.TestCase):
    def test_missing_calibration_is_saved_without_truth_or_parent_file_dependency(self):
        # A quiet event is also a real production edge case: do not force K lines,
        # invent time-to-distance calibration, or silently promote NaN to zero.
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);previous=directory/'observed.root'
            with StageWriter(previous,'ingest') as writer:
                writer.write(dict(entry=np.int64(7),file_index=np.int64(0),
                    source_group=np.int64(77),raw_waveform=np.zeros((12,64,512),np.float32),
                    si_edep_mev=np.empty(0),has_observed_si_edep=np.uint8(0)),
                    {'observed_only':True})
                # The order previously triggered an Uproot recursive-membership bug.
                writer.histogram('quiet_adc',[0.]);writer.metrics({'events':1.})
            config={'input_roots':['/missing/original.root'],'n_tracks':3,
                    'finding':{'k':3,'method':'hough','tb_bin':4},'drift':{},'masses':{}}
            for stage in ('prepare','finding','fitting','vertex','association','material_request','material_import','energy','kinematics'):
                output=directory/(stage+'.root')
                run_stage(stage,previous,output,config)
                previous.unlink()  # The new file must be sufficient for the next task.
                previous=output
            rows=list(read_stage(previous));self.assertEqual(len(rows),1)
            arrays,diagnostics=rows[0]
            self.assertFalse(arrays['kinematics_ok'])
            self.assertFalse(np.any(arrays['fit_ok']))
            self.assertTrue(np.isnan(arrays['invariant_mass_mev']))
            self.assertIn('missing_calibration',diagnostics['fitting']['status'])
            self.assertEqual(manifest(previous)['stage'],'kinematics')
            self.assertEqual(arrays['raw_waveform'].shape,(12,64,512))
            self.assertFalse(any(name.startswith('truth_') for name in arrays))


if __name__=='__main__':unittest.main()
