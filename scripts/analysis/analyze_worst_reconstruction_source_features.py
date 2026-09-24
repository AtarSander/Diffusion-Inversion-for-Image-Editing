#!/usr/bin/env python3
"""Compare source-image traits of worst reconstruction cases with valid PIE population."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
from PIL import Image

EDITORS=("pix2pix-zero","masactrl","pnp")
METRICS=("structure_distance","lpips_unedit_part","mse_unedit_part","psnr_unedit_part","ssim_unedit_part")
def finite(x):
    try: return math.isfinite(float(x))
    except (ValueError, TypeError): return False
def unedited_fraction(item):
    total=512*512; edited=0; rle=item["mask"]
    for start,length in zip(rle[::2],rle[1::2]): edited += max(0, min(length, total-start))
    edited += 2*512 + 2*(512-2) # same forced border as evaluation
    return max(0,total-edited)/total
def features(path):
    rgb=np.asarray(Image.open(path).convert("RGB").resize((256,256)),dtype=np.float32)/255
    gray=rgb @ np.array([.299,.587,.114],dtype=np.float32)
    hist=np.histogram(gray,bins=256,range=(0,1))[0].astype(np.float64); prob=hist/hist.sum(); prob=prob[prob>0]
    maxc=rgb.max(axis=2); minc=rgb.min(axis=2)
    return {
      "luminance_std":float(gray.std()),
      "mean_saturation":float(np.mean((maxc-minc)/np.maximum(maxc,1e-6))),
      "grayscale_entropy":float(-(prob*np.log2(prob)).sum()),
      "edge_energy":float((np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2),
    }
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--metrics-root',type=Path,required=True); ap.add_argument('--selection-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    mapping=json.loads((a.data/'mapping_file.json').read_text())
    results={}
    for editor in EDITORS:
      method='lora+directinversion+'+editor
      with (a.metrics_root/f'{method}.csv').open() as f: raw=list(csv.DictReader(f))
      valid=[r for r in raw if all(finite(r.get(f'{method}|{m}')) for m in METRICS)]
      with (a.selection_root/editor/'worst.csv').open() as f: worst_ids={r['file_id'] for r in csv.DictReader(f)}
      rows=[]
      for row in valid:
        item=mapping[row['file_id']]; path=a.data/'annotation_images'/item['image_path']
        row_features=features(path); row_features['unedited_fraction']=unedited_fraction(item)
        path_parts=item['image_path'].split('/')
        row_features['source_domain']='random' if path_parts[0].startswith('0_random') else path_parts[1]
        row_features['is_worst']=row['file_id'] in worst_ids; rows.append(row_features)
      worst=[r for r in rows if r['is_worst']]; rest=[r for r in rows if not r['is_worst']]
      item={"n_worst":len(worst),"n_rest":len(rest),"source_domain":{"worst":{},"rest":{}}}
      for label,group in (("worst",worst),("rest",rest)):
        for domain in ('1_artificial','2_natural','random'): item['source_domain'][label][domain]=sum(r['source_domain']==domain for r in group)/len(group)
      for key in ('luminance_std','mean_saturation','grayscale_entropy','edge_energy','unedited_fraction'):
        w=np.array([r[key] for r in worst]); rr=np.array([r[key] for r in rest]); pooled=np.sqrt((w.var()+rr.var())/2)
        item[key]={"worst_mean":float(w.mean()),"rest_mean":float(rr.mean()),"standardized_difference":float((w.mean()-rr.mean())/pooled) if pooled else 0.0}
      results[editor]=item
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
