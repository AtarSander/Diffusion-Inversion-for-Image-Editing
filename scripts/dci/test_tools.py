"""Tiny offline fixtures for archive verification, split preservation and safe extraction."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'data/raw/DCI'
        self.base.mkdir(parents=True)
        self.state = self.root / 'state'
        self.state.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.quota(10)
        self.env = dict(os.environ, DCI_PROJECT_DATA=str(self.root / 'data'),
                        DCI_ACQUISITION=str(self.state), DCI_QUOTA_MOUNT='/fixture',
                        PATH=str(self.bin) + os.pathsep + os.environ['PATH'])

    def quota(self, used):
        p = self.bin / 'lfs'
        p.write_text(f'#!/bin/sh\nprintf "/fixture 1 0 1000000 - {used} 0 10000 -\\n"\n')
        p.chmod(0o700)

    def archive(self, names):
        p = self.root / 'dci.tar.gz'
        with tarfile.open(p, 'w:gz') as stream:
            for name, payload in names.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        (self.state / 'annotation_download.json').write_text(json.dumps({
            'verified': True, 'archive_path': str(p),
            'expected_sha256': digest, 'actual_sha256': digest}))
        return p

    def run_tool(self, name):
        return subprocess.run([sys.executable, str(TOOLS / name)], env=self.env,
                              text=True, capture_output=True, timeout=30)

    def test_safe_extract(self):
        self.archive({'dci/hello.txt': b'hello'})
        result = self.run_tool('safe_extract_dci.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.base / 'dataset/dci/hello.txt').read_bytes(), b'hello')

    def test_traversal_rejected(self):
        self.archive({'../../outside': b'bad'})
        result = self.run_tool('safe_extract_dci.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe archive pathname', result.stderr)
        self.assertFalse((self.base / 'extracted.part').exists())

    def test_inode_quota_rejected(self):
        self.quota(9999)
        self.archive({'dci/a': b'a'})
        result = self.run_tool('safe_extract_dci.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Insufficient per-user inode quota', result.stderr)

    def test_validator_rechecks_digest(self):
        p = self.archive({'dci/a': b'a'})
        with p.open('ab') as stream:
            stream.write(b'changed')
        result = self.run_tool('validate_dci_archive.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Archive changed', result.stderr)

    def test_official_split_and_missing_image_preserved(self):
        sample = json.dumps({'image': 'sa_1.jpg', 'width': 8, 'height': 8,
                             'short_caption': 'fixture caption'}).encode()
        self.archive({'dci/annotations/1.json': sample, 'dci/complete/1.json': sample,
                      'dci/splits.json': b'{"train":[],"valid":[],"test":["1.json"]}'})
        result = self.run_tool('validate_dci_archive.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.state / 'dci_validation.json').read_text())
        self.assertEqual(report['official_splits']['test']['count'], 1)
        self.assertEqual(report['missing_unique_images'], 1)
        self.assertFalse(report['ready_for_reconstruction'])
        self.assertEqual(json.loads((self.base / 'metadata/splits.json').read_text())['test'], ['1.json'])


if __name__ == '__main__':
    unittest.main()
