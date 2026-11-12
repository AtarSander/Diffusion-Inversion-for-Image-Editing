import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).parents[1] /
                                             'scripts/analysis/summarize_ovg_seeds.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def fixture(self, root):
        cfg = dict(seed=42, sample_seed=42, lora={}, adapter_checkpoint='adapter.pt',
                   model_checkpoint={}, conditioning={}, data={}, inference={},
                   benchmark=dict(methods=[], mixed_precision='no', batch_size=1))
        dirs = []
        for i in range(4):
            d = root / str(i)
            d.mkdir()
            c = copy.deepcopy(cfg)
            c['seed'] += i
            result = dict(protocol='test', upstream_revision='abc', num_samples=2,
                          config=c, metrics={'method': {'psnr': 10+i, 'fid': 30-i}})
            (d / 'summary.json').write_text(json.dumps(result))
            (d / 'sample_manifest.csv').write_text('path\na.png\nb.png\n')
            dirs.append(d)
        return dirs

    def test_mean(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            summary.summarize(self.fixture(root), root / 'means')
            result = json.loads((root / 'means/mean_metrics.json').read_text())
            self.assertEqual(result['mean_metrics']['method'], {'psnr': 11.5, 'fid': 28.5})

    def test_mismatched_images_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            dirs = self.fixture(root)
            (dirs[-1] / 'sample_manifest.csv').write_text('path\na.png\nc.png\n')
            with self.assertRaisesRegex(ValueError, 'Image selections'):
                summary.summarize(dirs, root / 'means')

    def test_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            dirs = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                summary.summarize([dirs[0]]*4, root / 'means')


if __name__ == '__main__':
    unittest.main()
