"""CPU-only post-hoc summaries of immutable outline-reader outputs; no models/images."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def mean(xs):
    return statistics.mean(xs) if xs else None


def sem(xs):
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def validate_output_path(upstream, output):
    """Allow fresh sibling summaries, never overlap immutable run inputs."""
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    inputs = upstream / 'run'
    if output == upstream or output == inputs or inputs in output.parents or output in inputs.parents:
        raise ValueError('Diagnostics output overlaps immutable run inputs')
    if output.exists():
        raise FileExistsError(output)
    return upstream, output


def main(upstream, output):
    import torch
    torch.set_num_threads(2)
    upstream, output = validate_output_path(upstream, output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    def record(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
        hashes[str(p)] = h.hexdigest()
        return p
    bundles, scores, manifests = {}, {}, {}
    grid = {-16., -8., -3., 0., 3., 8., 16.}
    for arm in ('empty', 'different'):
        root = upstream / 'run' / arm
        bundles[arm] = torch.load(record(root / 'fit/directions.pt'), map_location='cpu', weights_only=True)
        files = list((root / 'edit').glob('scores_size_*.json'))
        if len(files) != 1: raise ValueError(f'{arm}: expected exactly one score JSON, found {len(files)}')
        scores[arm] = json.loads(record(files[0]).read_text())
        manifests[arm] = json.loads(record(root / 'edit/manifest.json').read_text())
    assert scores['empty']['threshold'] == scores['different']['threshold'] == 40.0
    assert scores['empty']['input_radius'] == scores['different']['input_radius']
    ids = set(scores['empty']['input_radius']); assert len(ids) == 8
    assert manifests['empty']['selected_layers'] == manifests['different']['selected_layers']
    assert manifests['empty']['active_calls'] == manifests['different']['active_calls']
    selected = set(manifests['empty']['selected_layers']); calls = manifests['empty']['active_calls']
    vectors = []
    da, db = bundles['empty']['directions'], bundles['different']['directions']
    assert set(da) == set(db)
    for rep in sorted(da):
        assert set(da[rep]) == set(db[rep])
        for method in sorted(da[rep]):
            assert set(da[rep][method]) == set(db[rep][method])
            for layer in sorted(da[rep][method]):
                x, y = (d[rep][method][layer]['true'].float() for d in (da, db))
                assert x.shape == y.shape and x.ndim == 2
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                for scope, steps in [('all_saved', list(range(x.shape[0]))), ('applied', calls if layer in selected and rep == 'SA' else [])]:
                    if not steps: continue
                    a, b = x[steps], y[steps]
                    nx, ny = a.norm(dim=1), b.norm(dim=1)
                    valid = (nx > 0) & (ny > 0)
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)[valid].clamp(-1, 1)
                    ratios = (ny[nx > 0] / nx[nx > 0]).tolist()
                    vectors.append(dict(representation=rep, method=method, layer=layer, scope=scope,
                        steps=len(steps), valid_angle_steps=int(valid.sum()), valid_ratio_steps=len(ratios),
                        cosine=mean(cos.tolist()), angle_degrees=mean((cos.acos() * (180 / math.pi)).tolist()),
                        empty_l2=float(nx.mean()), different_l2=float(ny.mean()), norm_ratio=mean(ratios)))
    indexed = {}
    for arm, data in scores.items():
        rows = {}
        for r in data['rows']:
            key = (r['method'], float(r['strength']), r['source_id'])
            if key in rows: raise ValueError(f'duplicate {arm} {key}')
            assert math.isfinite(r['ratio_vs_input'])
            rows[key] = r['ratio_vs_input']
        expected = {(m, s, i) for m in ('DDIM','MW') for s in grid for i in ids}
        if set(rows) != expected: raise ValueError(f'{arm}: incomplete/unexpected grid; observed {len(rows)}, expected {len(expected)}')
        indexed[arm] = rows
    curves = []
    for method in ('DDIM','MW'):
        for strength in sorted(grid):
            a = [indexed['empty'][method,strength,i] for i in sorted(ids)]
            b = [indexed['different'][method,strength,i] for i in sorted(ids)]
            delta = [y-x for x,y in zip(a,b)]
            curves.append(dict(method=method,strength=strength,n=8,empty_mean=mean(a),empty_sem=sem(a),
                different_mean=mean(b),different_sem=sem(b),paired_delta_mean=mean(delta),paired_delta_sem=sem(delta)))
    aggregates = []
    for rep, method, scope in sorted({(r['representation'],r['method'],r['scope']) for r in vectors}):
        rs=[r for r in vectors if (r['representation'],r['method'],r['scope'])==(rep,method,scope)]
        entry=dict(representation=rep,method=method,scope=scope,layers=len(rs))
        for metric in ('cosine','angle_degrees','empty_l2','different_l2','norm_ratio'):
            vals=[r[metric] for r in rs if r[metric] is not None]
            entry[metric]=mean(vals);entry[metric+'_valid_layers']=len(vals)
        aggregates.append(entry)
    limits=['No new evaluator, images, dataset, or model execution.',
        'Radius ratio uses per-image input radius then equal-weight mean across eight paired circles; SEM is descriptive, not a confidence interval.',
        'Vector statistics: per timestep, then equal timestep mean within layer, then equal layer mean; angles are mean acos(cosine), not acos(mean cosine). Zero-norm undefined cases are excluded and counted.',
        'all_saved covers every saved timestep/layer; applied restricts to SA selected layers and active_calls from the edit manifest.',
        'Reconstruction strength=0 radius ratio is the only existing preservation proxy used (ideal 1); no PSNR/SSIM/LPIPS is emitted by the inspected scorer.',
        'Thresholded area changes can reflect fading/sharpness instead of geometry. No universal higher-is-better direction applies to size response.',
        'Steering offsets are L2-normalized in the edit manifest; raw vector norm changes alone do not prove changed steering strength.',
        'Eight synthetic circles, one seed and unknown historical training-loss provenance limit generalization.']
    result=dict(upstream=str(upstream),input_sha256=hashes,threshold=40,n_sources=8,vector_aggregation=aggregates,
        vectors=vectors,response_curves=curves,preservation=[r for r in curves if r['strength']==0],limitations=limits)
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name, rows in [('vectors.csv',vectors),('response_curves.csv',curves)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Outline-reader diagnostics','',f'Paired sources: {len(ids)}; threshold: 40.','',
        '| method | strength | empty ratio | different ratio | paired delta | n |','|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['strength']:g} | {r['empty_mean']:.4f} | {r['different_mean']:.4f} | {r['paired_delta_mean']:.4f} | 8 |" for r in curves]
    lines += ['','Vector aggregation:','',json.dumps(aggregates,indent=2),'']+limits
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--upstream',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.upstream,args.output)
