"""Render twice with pinned non-ML dependencies, compare bytes, then preprocess once."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_sources(root):
    for family,count in [('triangles64_outline',128),('circles8_eval',8)]:
        manifest=root/family/'sources.jsonl'
        rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        assert len(rows)==count and len({r['id'] for r in rows})==count
        for row in rows:
            assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
            assert (manifest.parent/row['path']).is_file()


def main(output, historical):
    versions={n:importlib.metadata.version(n) for n in ('numpy','pillow','matplotlib','fire')}
    assert versions==dict(numpy='2.4.4',pillow='12.2.0',matplotlib='3.11.2',fire='0.7.1'),versions
    from PIL import Image,ImageOps
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]; commands=[]
    old=Path(historical).resolve();verify_sources(old)
    old_hash={family:inventory(old/family) for family in ('triangles64_outline','circles8_eval')}
    for repeat in ('render_a','render_b'):
        dest=root/repeat;dest.mkdir()
        for script,family,args in [
            ('make_triangle_dataset.py','triangles64_outline',['--pairs=64','--big_radius=300','--small_radius=100','--seed=0','--outline=True','--ink_area=10000']),
            ('make_circle_eval_set.py','circles8_eval',['--count=8','--radius=175','--grow_room=300','--seed=1'])]:
            command=[sys.executable,str(repo/'scripts'/script),'--out_dir='+str(dest/family),*args]
            commands.append(command);subprocess.run(command,check=True)
        verify_sources(dest)
    a,b=inventory(root/'render_a'),inventory(root/'render_b')
    assert a==b,'Regeneration differs across distinct roots'
    for render_name, output_name in (("render_a","inputs512"),("render_b","verify512")):
        normalized=root/output_name;normalized.mkdir()
        for family in ('triangles64_outline','circles8_eval'):
            src=root/render_name/family;dst=normalized/family;dst.mkdir()
            (dst/'sources.jsonl').write_bytes((src/'sources.jsonl').read_bytes())
            for row in map(json.loads,(src/'sources.jsonl').read_text().splitlines()):
                with Image.open(src/row['path']) as image:
                    image=ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'),(512,512),
                        method=Image.Resampling.LANCZOS,centering=(.5,.5))
                    image.save(dst/row['path'],format='PNG')
            meta=json.loads((src/'dataset_meta.json').read_text());meta['preprocessing']=dict(source_size=1024,size=512,mode='RGB',resampling='LANCZOS',centering=[.5,.5],exif_transpose=True)
            (dst/'dataset_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        verify_sources(normalized)
    assert inventory(root/'inputs512') == inventory(root/'verify512'), '512 preprocessing byte mismatch'
    normalized=root/'inputs512'
    report=dict(python=sys.version,versions=versions,backend='Agg',commands=commands,
        generation_twice_byte_identical=True,preprocessing_twice_byte_identical=True,files_per_render=len(a),render_sha256=a,
        normalized_sha256=inventory(normalized),historical_sha256=old_hash,
        historical_match={f:old_hash[f]==inventory(root/'render_a'/f) for f in old_hash},
        notes='Raw 1024 geometry retained, deterministic 512 LANCZOS input for both model and scorer. No model imports.')
    (root/'input_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('render_sha256','normalized_sha256','historical_sha256','commands')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--historical',required=True);a=p.parse_args();main(a.output,a.historical)
