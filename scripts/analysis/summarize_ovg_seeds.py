"""Validate comparable seed runs and report arithmetic means, not pooled FID."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def signature(summary):
    cfg = summary['config']
    return dict(protocol=summary['protocol'], revision=summary['upstream_revision'],
        num_samples=summary['num_samples'], sample_seed=cfg.get('sample_seed', cfg['seed']),
        **{k: cfg[k] for k in ('lora', 'adapter_checkpoint', 'model_checkpoint', 'conditioning', 'data', 'inference')},
        methods=cfg['benchmark']['methods'], precision=cfg['benchmark']['mixed_precision'],
        batch_size=cfg['benchmark']['batch_size'])


def summarize(directories, output):
    summaries = [json.loads((d / 'summary.json').read_text()) for d in directories]
    seeds = [s['config']['seed'] for s in summaries]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError(f'Expected four distinct completed seeds, found {seeds}')
    reference = signature(summaries[0])
    if any(signature(s) != reference for s in summaries):
        raise ValueError('Model, dataset, sample seed, precision or method settings differ')
    selections = []
    for d, s in zip(directories, summaries):
        with (d / 'sample_manifest.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != s['num_samples']:
            raise ValueError(f'Incomplete manifest: {d}')
        selections.append([r.get('path', r.get('image_path')) for r in rows])
    if any(x != selections[0] for x in selections):
        raise ValueError('Image selections/order differ across seeds')
    methods = set(summaries[0]['metrics'])
    if any(set(s['metrics']) != methods for s in summaries):
        raise ValueError('Missing methods in a seed run')
    means = {}
    for method in sorted(methods):
        keys = set(summaries[0]['metrics'][method])
        if any(set(s['metrics'][method]) != keys for s in summaries):
            raise ValueError(f'Metric coverage mismatch: {method}')
        means[method] = {}
        for metric in sorted(keys):
            values = [s['metrics'][method][metric] for s in summaries]
            if isinstance(values[0], bool):
                continue
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError(f'Non-finite/missing metric: {method}/{metric}')
            # An invalid kNN denominator is not a successful recovery measurement.
            if metric.startswith('knn_recovery') and not all(
                s['metrics'][method].get('knn_recovery_valid', True) for s in summaries):
                continue
            means[method][metric] = statistics.mean(values)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(seeds=seeds, images_per_seed=reference['num_samples'], mean_metrics=means,
        aggregation='arithmetic mean of per-seed metrics; FID is not pooled',
        source_summaries=[str(d / 'summary.json') for d in directories],
        source_sha256=[hashlib.sha256((d / 'summary.json').read_bytes()).hexdigest() for d in directories],
        protocol=reference, per_seed={str(seed): s['metrics'] for seed, s in zip(seeds, summaries)})
    (output / 'mean_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = [dict(method=m, metric=k, mean=v) for m, vals in means.items() for k, v in vals.items()]
    with (output / 'mean_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'metric', 'mean'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('run_dirs', type=Path, nargs=4)
    args = parser.parse_args()
    summarize(args.run_dirs, args.output_dir)
