"""Stdlib-only checks for non-overwriting delivery into an existing checkout.

Run: python3 checks/check_install_new_macros.py
This checks file preservation, not ROOT or detector physics.
"""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile
import unittest

MACROS = Path(__file__).resolve().parents[1]


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}


class InstallChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hoyle delivery ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'downloaded checkout' / 'hoyle_prepare_20261027'
        shutil.copytree(MACROS, self.source)
        self.repo = self.root / 'existing server checkout'
        for name, data in {
            'tpcdrum/simulation/SimParameters.mac': b'GasPressure 650\nGEM unchanged\n',
            'tpcdrum/simulation/makeSimData.C': b'existing locally edited macro\n',
            'tpcdrum/task/STDSiEdepSumTask.cpp': b'existing server task\n',
            'data/C12_existing_MC.root': b'existing MC sentinel\x00\x01',
            'data/C12_existing_digi.root': b'existing digi sentinel\x00\x02',
            '.git/config': b'existing repository configuration\n',
            '.gitignore': b'*.root\n',
        }.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.destination = self.repo / 'tpcdrum/simulation/hoyle_prepare_20261027'

    def run_install(self):
        return subprocess.run(['bash', str(self.source / 'install_new_macros.sh'), str(self.repo)],
                              capture_output=True, text=True)

    def test_adds_only_new_folder_and_preserves_existing_bytes(self):
        before, source_before = hashes(self.repo), hashes(self.source)
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        after = hashes(self.repo)
        for name, digest in before.items():
            self.assertEqual(after[name], digest, name)
        added = set(after) - set(before)
        self.assertTrue(added)
        self.assertTrue(all(n.startswith('tpcdrum/simulation/hoyle_prepare_20261027/') for n in added))
        self.assertEqual(hashes(self.destination), source_before)
        self.assertEqual(hashes(self.source), source_before)

    def test_existing_destination_refused_without_changes(self):
        self.destination.mkdir()
        (self.destination / 'local_work.C').write_text('keep this local work')
        before = hashes(self.repo)
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(hashes(self.repo), before)

    def test_symlink_destination_refused_without_changes(self):
        actual = self.root / 'other local folder'
        actual.mkdir()
        (actual / 'keep.root').write_bytes(b'keep untouched')
        self.destination.symlink_to(actual, target_is_directory=True)
        before, actual_before = hashes(self.repo), hashes(actual)
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(hashes(self.repo), before)
        self.assertEqual(hashes(actual), actual_before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
