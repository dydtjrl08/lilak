"""Export-driver preparation/collection checks; no Geant4 execution claimed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

CAL = Path(__file__).resolve().parents[1]/'calibration'


def module(name):
    spec = importlib.util.spec_from_file_location(name, CAL/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


build = module('export_build_run')
collect = module('export_collect_inputs')


class ExportPreparationTests(unittest.TestCase):
    def test_reuses_build_flags_but_never_overwrites_original_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            directory = repo/'build/CMakeFiles/run_sim.exe.dir'
            directory.mkdir(parents=True)
            (directory/'flags.make').write_text('CXX_DEFINES = -DREAL_CONFIG=1\nCXX_INCLUDES = -I/source/include\nCXX_FLAGS = -std=c++14 -O2\n')
            original = '/usr/bin/c++ CMakeFiles/run_sim.exe.dir/tpcdrum/simulation/run_sim.cc.o other.o -o /old/run_sim.exe -lLILAK -lNPTOOLSIM'
            (directory/'link.txt').write_text(original)
            output = repo/'new exporter'
            output.mkdir()
            compile_cmd, link, exe, receipt = build.commands(repo, repo/'build', None, output)
            self.assertIn('-DREAL_CONFIG=1', compile_cmd)
            self.assertIn('-I/source/include', compile_cmd)
            self.assertIn('other.o', link)
            self.assertNotIn('CMakeFiles/run_sim.exe.dir/tpcdrum/simulation/run_sim.cc.o', link)
            self.assertEqual(link[link.index('-o')+1], str(exe))
            self.assertTrue(str(exe).startswith(str(output)))
            self.assertEqual((directory/'link.txt').read_text(), original)
            self.assertFalse(receipt['original_build_modified'])

    def test_exact_parameter_asset_path_and_manifest_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)/'repository with spaces'
            simulation = repo/'tpcdrum/simulation'
            simulation.mkdir(parents=True)
            assets = repo/'actual assets'
            assets.mkdir()
            (assets/'gas.root').write_bytes(b'fixture asset bytes')
            parameter = simulation/'SimParameters.mac'
            text = 'TPCDrum/\n  GasPressure 650. # comment\n  SimDataPath "'+str(assets)+'"\n  STDGarfieldData gas.root\n  PulseShapeData missing.root\nNPTool/\n  ReactionFile direct.reaction\n'
            parameter.write_text(text)
            (simulation/'direct.reaction').write_text('fixture reaction')
            parsed = collect.parse_parameters(parameter)
            self.assertEqual(parsed['TPCDrum/GasPressure'], '650.')
            self.assertEqual(parsed['TPCDrum/SimDataPath'], str(assets))
            out = Path(temporary)/'new collection'
            # No ROOT is needed for bounded plain-file evidence collection.
            import unittest.mock as mock
            with mock.patch.object(collect.shutil, 'which', return_value=None):
                self.assertEqual(collect.main(['--repo', str(repo), '--output', str(out)]), 0)
            manifest = json.loads((out/'collection_manifest.json').read_text())
            self.assertFalse(manifest['raw_digi_copied'])
            self.assertFalse(manifest['physical_calibration_validated'])
            self.assertTrue(manifest['collection_has_missing_inputs'])
            self.assertEqual((out/'assets/gas.root').read_bytes(), b'fixture asset bytes')
            self.assertEqual(parameter.read_text(), text)
            self.assertTrue(out.with_suffix('.tar.gz').is_file())
            with self.assertRaises(FileExistsError):
                collect.main(['--repo', str(repo), '--output', str(out)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
