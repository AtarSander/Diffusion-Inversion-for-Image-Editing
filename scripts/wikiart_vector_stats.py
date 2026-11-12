"""Secondary saved-vector diagnostics, same aggregation as outline diagnostics."""
import argparse,json,os,statistics
from pathlib import Path

def main(run,output):
    assert os.environ.get('SLURM_JOB_ID')
    import torch
    run=Path(run)
    bundles=[torch.load(run/a/'fit/directions.pt',map_location='cpu',weights_only=True)['directions'] for a in ('empty','different')]
    manifest=json.loads((run/'empty/edit/manifest.json').read_text());layers=manifest['selected_layers'];calls=manifest['active_calls'];rows=[]
    for method in ('DDIM','MW'):
        for layer in layers:
            x,y=[d['SA'][method][layer]['true'].float()[calls] for d in bundles]
            assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
            nx,ny=x.norm(dim=1),y.norm(dim=1);valid=(nx>0)&(ny>0)
            cos=torch.nn.functional.cosine_similarity(x,y,dim=1)[valid].clamp(-1,1)
            rows.append(dict(method=method,layer=layer,valid_calls=int(valid.sum()),zero_norm_calls=int((~valid).sum()),cosine=float(cos.mean()) if len(cos) else None,angle_degrees=float(torch.rad2deg(cos.acos()).mean()) if len(cos) else None,empty_norm=float(nx.mean()),different_norm=float(ny.mean())))
    summary=[]
    for m in ('DDIM','MW'):
        rs=[r for r in rows if r['method']==m];entry={'method':m}
        for key in ('cosine','angle_degrees','empty_norm','different_norm'):
            v=[r[key] for r in rs if r[key] is not None];entry[key]=statistics.mean(v) if v else None
        summary.append(entry)
    Path(output).write_text(json.dumps(dict(layers=rows,summary=summary,aggregation='per call then equal-weight layer means; zero-norm angles excluded and counted'),allow_nan=False,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.run,a.output)
