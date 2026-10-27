"""Independent ROOT TLorentzVector reference for NumPy relativistic kinematics."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruction.algorithms.physics import kinematics

ROOT = os.environ.get('ROOT_EXECUTABLE', shutil.which('root'))


@unittest.skipUnless(ROOT, 'Set ROOT_EXECUTABLE for independent native kinematics check')
class RootKinematics(unittest.TestCase):
    def test_boost_mass_and_angles_against_root(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            macro, output = folder/'root_reference.C', folder/'reference.json'
            macro.write_text('''
void root_reference(const char* output) {
  // Synthetic software fixture, not simulated or measured detector data.
  const double m=3727.3794066, ground=11174.86;
  double momentum[3][3]={{20.,275.,10.},{30.,300.,-12.},{-10.,260.,25.}};
  TLorentzVector a[3], parent;
  for (int i=0;i<3;++i) {
    a[i].SetXYZM(momentum[i][0],momentum[i][1],momentum[i][2],m);parent+=a[i];
  }
  FILE* out=fopen(output,"wx");
  fprintf(out,"{\\\"mass\\\":%.17g,\\\"excitation\\\":%.17g,\\\"com\\\":[",parent.M(),parent.M()-ground);
  TVector3 b=parent.BoostVector();
  for (int i=0;i<3;++i) {
    a[i].Boost(-b);
    if(i)fprintf(out,",");
    fprintf(out,"[%.17g,%.17g,%.17g,%.17g]",a[i].Px(),a[i].Py(),a[i].Pz(),a[i].E());
  }
  fprintf(out,"],\\\"angles\\\":[%.17g,%.17g,%.17g]}\\n",
    a[0].Angle(a[1].Vect())*TMath::RadToDeg(),
    a[0].Angle(a[2].Vect())*TMath::RadToDeg(),
    a[1].Angle(a[2].Vect())*TMath::RadToDeg());fclose(out);
}
''')
            done = subprocess.run([ROOT, '-n', '-l', '-b', '-q',
                '%s(%s)' % (macro, json.dumps(str(output)))], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
            self.assertEqual(done.returncode, 0, done.stdout[-6000:])
            reference = json.loads(output.read_text())
            momentum = np.array([[20., 275., 10.], [30., 300., -12.], [-10., 260., 25.]])
            vectors = np.column_stack([momentum, np.sqrt(3727.3794066**2+(momentum**2).sum(axis=1))])
            result = kinematics(vectors, 11174.86)
            self.assertAlmostEqual(result['invariant_mass_mev'], reference['mass'], places=9)
            self.assertAlmostEqual(result['excitation_mev'], reference['excitation'], places=9)
            np.testing.assert_allclose(result['com_four_vectors'], reference['com'], atol=1e-10)
            np.testing.assert_allclose(result['com_opening_angles_deg'], reference['angles'], atol=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
