from pathlib import Path
import tempfile
import unittest
from reconstruction.install_new_reconstruction import install


class NewPathInstallerChecks(unittest.TestCase):
    def test_new_paths_only_existing_files_byte_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)/'repo';simulation=repo/'tpcdrum/simulation';simulation.mkdir(parents=True)
            old=simulation/'old_user_macro.C';old.write_bytes(b'original-user-content')
            outputs=install(repo)
            self.assertEqual(len(outputs),3)
            self.assertTrue((repo/'tpcdrum/reconstruction/run.py').is_file())
            self.assertTrue((simulation/'C12_run_reconstruction.C').is_file())
            self.assertEqual(old.read_bytes(),b'original-user-content')
            with self.assertRaises(FileExistsError):install(repo)
            self.assertEqual(old.read_bytes(),b'original-user-content')

    def test_all_destination_checks_precede_copy_and_symlinks_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)/'repo';simulation=repo/'tpcdrum/simulation';simulation.mkdir(parents=True)
            target=Path(folder)/'protected';target.write_bytes(b'keep')
            (simulation/'C12_view_reconstruction.C').symlink_to(target)
            with self.assertRaises(FileExistsError):install(repo)
            self.assertFalse((repo/'tpcdrum/reconstruction').exists())
            self.assertFalse((simulation/'C12_run_reconstruction.C').exists())
            self.assertTrue((simulation/'C12_view_reconstruction.C').is_symlink())
            self.assertEqual(target.read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
