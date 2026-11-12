"""Paired image-level summaries, shared bootstrap draws, fixed overview grids."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def summarize(rows,expected):
    ids=sorted({r['source_id'] for r in rows});assert len(ids)==expected
    ix={(r['source_id'],r['method'],r['arm'],float(r['strength'])):r for r in rows}
    assert len(ix)==len(rows)==12*expected
    draws=np.random.default_rng(20261103).integers(0,expected,size=(2000,expected))
    metrics=['csd_target','csd_gain','dino_input','dino_recon','ssim_input','ssim_recon','mse_input','mse_recon','psnr_input','psnr_recon']
    def stats(values):
        if any(v is None for v in values):
            return dict(mean=None,sem=None,ci95=None,n=expected,undefined_count=sum(v is None for v in values),reason='Exact matches have infinite PSNR; no finite aggregate reported')
        v=np.asarray(values,dtype=float);assert np.isfinite(v).all()
        return dict(mean=float(v.mean()),sem=float(v.std(ddof=1)/math.sqrt(expected)),ci95=np.quantile(v[draws].mean(axis=1),[.025,.975]).tolist(),n=expected)
    out=[]
    for method in ('DDIM','MW'):
        for strength in (0.,1.,2.):
            for metric in metrics:
                a=[ix[i,method,'empty',strength][metric] for i in ids]
                b=[ix[i,method,'different',strength][metric] for i in ids]
                out.append(dict(method=method,strength=strength,metric=metric,empty=stats(a),different=stats(b),paired_difference=stats([y-x if x is not None and y is not None else None for x,y in zip(a,b)])))
    return dict(n=expected,bootstrap_draws=2000,bootstrap_seed=20261103,shared_resamples=True,rows=out,limitations=['Intervals condition on one construction/reference split and one seed.','Between-method comparisons are end-to-end systems with different vectors.','Pixel similarity may decrease for legitimate style changes; no combined score.'])

def main(run,expected):
    run=Path(run);q=json.loads((run/'quality.json').read_text());rows=q['rows'];result=summarize(rows,expected)
    output=run.parent/'comparison';output.mkdir(exist_ok=False)
    (output/'quality_summary.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    with (output/'per_image.csv').open('w') as f:
        keys=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    text=['# WikiArt paired quality','',f'n={expected} paired source images per method/strength; 2000 shared image-bootstrap draws.','', 'CSD gain is relative to the same image/method reconstruction. DINO, SSIM and PSNR compare against input and reconstruction. LPIPS is not enabled.','', '|method|strength|CSD gain empty|CSD gain styled|paired difference [95% CI]|','|---|---|---|---|---|']
    for r in result['rows']:
        if r['metric']=='csd_gain':text.append(f"|{r['method']}|{r['strength']}|{r['empty']['mean']:.5f}|{r['different']['mean']:.5f}|{r['paired_difference']['mean']:.5f} {r['paired_difference']['ci95']}|")
    (output/'summary.md').write_text('\n'.join(text)+'\n')
    # Fixed lexicographic first four source IDs, chosen independently of metrics.
    from PIL import Image,ImageDraw
    ids=sorted({r['source_id'] for r in rows})[:4]
    ix={(r['source_id'],r['method'],r['arm'],float(r['strength'])):r for r in rows}
    for method in ('DDIM','MW'):
        grid=Image.new('RGB',(6*192,len(ids)*220),'white');draw=ImageDraw.Draw(grid)
        for y,i in enumerate(ids):
            base=ix[i,method,'empty',0.]
            panels=[('input',base['source_path']),('recon',base['image_path'])]+[(f'{arm} a={a}',ix[i,method,arm,a]['image_path']) for arm in ('empty','different') for a in (1.,2.)]
            for x,(label,path) in enumerate(panels):
                grid.paste(Image.open(path).convert('RGB').resize((192,192)),(x*192,y*220));draw.text((x*192,y*220+193),label,fill='black')
        grid.save(output/f'overview_{method}.png')
    (output/'overview_ids.json').write_text(json.dumps(ids)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--expected',type=int,required=True);a=p.parse_args();main(a.run,a.expected)
