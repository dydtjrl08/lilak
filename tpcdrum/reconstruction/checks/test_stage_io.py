import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from reconstruction.stage_io import StageWriter,read_stage,manifest


class StageChecks(unittest.TestCase):
    def test_full_arrays_survive_independent_root_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'stage.root'
            wave=np.arange(12*64*512,dtype=np.float32).reshape(12,64,512)
            with StageWriter(p,'prepared',{'role':'observations'}) as writer:
                for i in range(2):
                    writer.write({'entry':np.int64(i),'file_index':np.int64(0),
                                  'waveform':wave+i,'empty':np.empty((0,3)),
                                  'valid':np.asarray(i==0)}, {'meaning':'observations only'})
                writer.metrics({'charge_sum':float(wave.sum())})
                writer.histogram('waveform_samples',wave,bins=32)
            self.assertEqual(manifest(p)['events'],2)
            rows=list(read_stage(p))
            for i,(row,diagnostic) in enumerate(rows):
                np.testing.assert_array_equal(row['waveform'],wave+i)
                self.assertEqual(row['empty'].shape,(0,3))
                self.assertEqual(int(row['entry']),i)
                self.assertEqual(diagnostic['meaning'],'observations only')
            import uproot
            with uproot.open(p) as f:
                self.assertEqual(f['events'].classname,'TTree')
                self.assertEqual(f['metrics'].num_entries,1)
                self.assertIn('performance/waveform_samples',f)
            self.assertEqual(len(list(Path(folder).glob('*'))),1)

    def test_abort_duplicate_truth_and_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'existing.root';p.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):StageWriter(p,'x')
            self.assertEqual(p.read_bytes(),b'existing')
            output=Path(folder)/'abort.root'
            row={'entry':np.int64(0),'file_index':np.int64(0),'data':np.ones(3)}
            with self.assertRaises(ValueError):
                with StageWriter(output,'x') as writer:
                    writer.write(row);writer.write(row)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                with StageWriter(output,'x') as writer:
                    writer.write(dict(row,truth_theta=np.zeros(3)))
            self.assertFalse(output.exists())
            truth=Path(folder)/'truth.root'
            with StageWriter(truth,'truth',{'role':'truth'}) as writer:
                writer.write(dict(row,truth_theta=np.zeros(3)))
            with self.assertRaises(ValueError):list(read_stage(truth))
            self.assertEqual(len(list(read_stage(truth,allow_truth=True))),1)


if __name__=='__main__':unittest.main()
