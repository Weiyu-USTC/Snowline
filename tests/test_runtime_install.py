"""Runtime release installation must reject incomplete or altered bundles."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('runtime_install', Path(__file__).resolve().parents[1]/'scripts/install_runtime.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class RuntimeInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'artifacts').mkdir()
        self.archive = self.root/'bundle.zip'
        self.files = {'outputs/models/test.bin': b'verified model', 'data/reference.csv': b'date,value\n2020,1\n'}
        self.manifest(self.files)

    def manifest(self, files):
        rows = [{'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} for name, body in files.items()]
        (self.root/'artifacts/runtime-files.json').write_text(json.dumps({'files': rows}))

    def bundle(self, files):
        with zipfile.ZipFile(self.archive, 'w') as zipped:
            for name, body in files.items():
                zipped.writestr(name, body)

    def test_verified_bundle_installs_and_is_idempotent(self):
        self.bundle(self.files)
        self.assertEqual(installer.install(self.archive, self.root), 2)
        self.assertEqual(installer.install(self.archive, self.root), 2)
        for name, body in self.files.items():
            self.assertEqual((self.root/name).read_bytes(), body)

    def test_bad_hash_changes_no_installed_file(self):
        self.bundle({**self.files, 'data/reference.csv': b'X'*len(self.files['data/reference.csv'])})
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            installer.install(self.archive, self.root)
        self.assertFalse((self.root/'outputs/models/test.bin').exists())

    def test_missing_or_extra_members_are_rejected(self):
        for files in [dict(list(self.files.items())[:1]), {**self.files, '../outside': b'no'}]:
            with self.subTest(names=list(files)):
                self.bundle(files)
                with self.assertRaisesRegex(ValueError, 'contents differ'):
                    installer.install(self.archive, self.root)

    def test_modified_existing_file_requires_explicit_replace(self):
        self.bundle(self.files)
        target = self.root/'data/reference.csv'
        target.parent.mkdir()
        target.write_bytes(b'local modification')
        with self.assertRaisesRegex(ValueError, 'Existing file differs'):
            installer.install(self.archive, self.root)
        self.assertEqual(target.read_bytes(), b'local modification')
        installer.install(self.archive, self.root, replace=True)
        self.assertEqual(target.read_bytes(), self.files['data/reference.csv'])

    def test_bad_manifest_path_and_symlink_are_rejected(self):
        for name in ['../outside', '/tmp/outside', 'src/overwritten.py', 'data/../outside']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                installer.safe_target(self.root, name)
        (self.root/'data').symlink_to(self.root/'artifacts', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symbolic links'):
            installer.safe_target(self.root, 'data/reference.csv')


if __name__ == '__main__':
    unittest.main()
