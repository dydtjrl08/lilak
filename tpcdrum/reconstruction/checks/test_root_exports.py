"""ROOT-only fixtures test real file/graph/template exports; no detector labels."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
ROOT = os.environ.get('ROOT_EXECUTABLE', shutil.which('root'))


@unittest.skipUnless(ROOT, 'Set ROOT_EXECUTABLE to test the native ROOT exports')
class RootExports(unittest.TestCase):
    def run_root(self, expression):
        result = subprocess.run([ROOT, '-n', '-l', '-b', '-q', expression], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout[-6000:])

    def test_graph_template_and_header_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture = folder/'create_fixture.C'
            fixture.write_text('''
void create_fixture(const char* folder) {
  { TFile f(TString(folder)+"/gas.root","RECREATE");
    double x[]={100.,200.}, y[]={.008,.009};
    for (const char* name:{"DriftVelocity","TranseverseDiffusion","LongitudinalDiffusion"}) {
      TGraph g(2,x,y);g.SetName(name);g.SetTitle("synthetic software fixture");g.Write();
    }
  }
  { TFile f(TString(folder)+"/pulse.root","RECREATE");
    TTree pulse("pulse","synthetic software fixture"); double value[300];
    for (int i=0;i<300;++i) value[i]=i*.25;
    pulse.Branch("template",value,"template[300]/D");pulse.Fill();pulse.Write();
  }
  { TFile f(TString(folder)+"/events.root","RECREATE");
    TTree event("event","synthetic software fixture");int counter=3;
    event.Branch("counter",&counter,"counter/I");event.Fill();event.Write();
  }
}
''')
            self.run_root('%s(%s)' % (fixture, json.dumps(str(folder))))
            output = folder/'transport.json'
            self.run_root('%s(%s,%s,%s)' % (
                BASE/'calibration/export_transport_assets.C',
                json.dumps(str(folder/'gas.root')), json.dumps(str(folder/'pulse.root')),
                json.dumps(str(output))))
            obj = json.loads(output.read_text())
            self.assertEqual(len(obj['graphs']), 3)
            self.assertEqual(obj['graphs'][0]['x'], [100., 200.])
            self.assertEqual(obj['graphs'][0]['y'], [.008, .009])
            self.assertEqual(len(obj['pulse_template']), 300)
            self.assertEqual(obj['pulse_template'][-1], 74.75)
            self.assertFalse(obj['drift_time_calibrated_for_charge_mean'])
            header = folder/'header.json'
            self.run_root('%s(%s,%s)' % (
                BASE/'calibration/export_file_header.C', json.dumps(str(folder/'events.root')),
                json.dumps(str(header))))
            obj = json.loads(header.read_text())
            self.assertEqual(obj['entries'], 1)
            self.assertEqual(obj['branches'][0]['name'], 'counter')
            self.assertFalse(obj['waveforms_read'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
