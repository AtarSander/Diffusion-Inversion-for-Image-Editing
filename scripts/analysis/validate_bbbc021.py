"""Fail if a BBBC021 run has incomplete metrics, exports, or an inactive LoRA."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def validate(root):
    summary = json.loads((root / 'summary.json').read_text())
    n = summary['config']['benchmark']['num_samples']
    assert summary['num_samples'] == n, 'Incomplete sample count'
    expected = {f'{m["name"]}_config_{i}' for i, m in
                enumerate(summary['config']['benchmark']['methods']) if m.get('enabled', True)}
    assert set(summary['metrics']) == expected, 'Missing method results'
    paths = {}
    for file in (root / 'images').rglob('*.h5'):
        with h5py.File(file, 'r') as f:
            name = f['config'].attrs['method']
            assert name not in paths, f'Duplicate export for {name}'
            paths[name] = file
            samples = sorted(k for k in f if k.startswith('sample_'))
            assert len(samples) == n, (name, len(samples), n)
            assert 'metrics' in f and 'timing' in f, f'Unfinished export: {name}'
            for key in samples:
                for field in ('generated_hr', 'original_hr', 'input_lr', 'latents'):
                    a = f[key][field][...]
                    assert np.isfinite(a).all(), (name, key, field)
    assert set(paths) == expected, 'Missing image exports'
    for name, metrics in summary['metrics'].items():
        for key in ('psnr (↑)', 'msssim (↑)', 'lpips (↓)', 'fid (↓)'):
            assert np.isfinite(metrics[key]), (name, key, metrics[key])
    matched = next((k for k in paths if k.startswith('matched_ddim_')), None)
    lora = next((k for k in paths if k.startswith('lora_inversion_')), None)
    if matched and lora:
        with h5py.File(paths[matched]) as a, h5py.File(paths[lora]) as b:
            keys = sorted(k for k in a if k.startswith('sample_'))
            changed = False
            for key in keys:
                for field in ('original_hr', 'input_lr'):
                    np.testing.assert_array_equal(a[key][field][...], b[key][field][...])
                changed |= not np.array_equal(a[key]['latents'][...], b[key]['latents'][...])
            assert changed, 'LoRA and matched-DDIM endpoints are identical'
    print(f'PASS: {len(expected)} methods x {n} images, finite metrics/exports, LoRA changes inversion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output_dir', type=Path)
    validate(parser.parse_args().output_dir)
