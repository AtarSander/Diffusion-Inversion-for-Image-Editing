"""Real-edit manifest adapter for existing CSD and PSNR/SSIM evaluators.
Only called inside the allocated job. No training, learned probe, or CLIP proxy.
"""
import argparse, gc, hashlib, json, math, os
from pathlib import Path

def load_csd_backbone(vision, converted):
    """Transformers4 wraps vision_model; Transformers5 exposes the tower directly.
    Preserve strict full-key/shape loading in either layout; no weight coercion.
    """
    tower = getattr(vision, 'vision_model', vision)
    return tower.load_state_dict(converted, strict=True)

def main(run, data, assets, expected):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Allocated job required')
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
    from diff_inversion.steering.csd import _convert_backbone, CSDModel, csd_style_features, mean_ref_similarity
    from diff_inversion.eval.alignment import calculate_psnr, calculate_ssim
    run, data, assets = map(Path, (run,data,assets))
    rows=[]
    for arm in ('empty','different'):
        manifest=json.loads((run/arm/'edit/manifest.json').read_text())
        for r in manifest['images']:
            rows.append(dict(r,arm=arm,image_path=str(run/arm/'edit'/r['path'])))
    ids=sorted({r['source_id'] for r in rows})
    assert len(ids)==expected
    indexed={(r['source_id'],r['method'],r['arm'],float(r['strength'])):r for r in rows}
    assert len(indexed)==len(rows)==expected*12
    assert set(indexed)=={(i,m,a,s) for i in ids for m in ('DDIM','MW') for a in ('empty','different') for s in (0.,1.,2.)}
    for i in ids:
        for m in ('DDIM','MW'):
            paths=[Path(indexed[i,m,a,0.]['image_path']) for a in ('empty','different')]
            assert hashlib.sha256(paths[0].read_bytes()).digest()==hashlib.sha256(paths[1].read_bytes()).digest(), 'Unpaired reconstruction'
    refs=[json.loads(l) for l in (data/'reference.jsonl').read_text().splitlines()]
    assert len(refs)==32
    refpaths=[str(data/r['path']) for r in refs]
    paths=list(dict.fromkeys([r['image_path'] for r in rows]+[r['source_path'] for r in rows]+refpaths))
    device='cuda'
    # Reuse the existing restricted CSD checkpoint semantics, with pinned local bytes.
    import argparse as ap
    allow=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,ap.Namespace]
    with torch.serialization.safe_globals(allow):
        checkpoint=torch.load(assets/'csd/pytorch_model.bin',map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in checkpoint['model_state_dict'].items()}
    projection=state.pop('last_layer_style');state.pop('last_layer_content')
    backbone={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
    assert len(backbone)==len(state)
    cfg=CLIPVisionConfig.from_pretrained(assets/'csd_processor',local_files_only=True)
    vision=CLIPVisionModel(cfg)
    # Existing converter returns keys relative to vision_model, not its wrapper.
    load_csd_backbone(vision, _convert_backbone(backbone))
    model=CSDModel(vision,projection).to(device).eval()
    processor=CLIPImageProcessor.from_pretrained(assets/'csd_processor',local_files_only=True)
    del checkpoint,state,backbone,vision;gc.collect()
    features={}
    for start in range(0,len(paths),8):
        subset=paths[start:start+8]
        imgs=[Image.open(p).convert('RGB') for p in subset]
        vals=csd_style_features(imgs,processor,model,8,device)
        features.update(zip(subset,vals))
    references=torch.stack([features[p] for p in refpaths])
    for r in rows:r['csd_target']=float(mean_ref_similarity(features[r['image_path']][None],references)[0])
    del features,references,model,processor;gc.collect();torch.cuda.empty_cache()
    processor=AutoImageProcessor.from_pretrained(assets/'dino',local_files_only=True)
    model=AutoModel.from_pretrained(assets/'dino',local_files_only=True,use_safetensors=True).to(device).eval()
    features={}
    with torch.no_grad():
        for start in range(0,len(paths),8):
            subset=paths[start:start+8]
            tensors=processor(images=[Image.open(p).convert('RGB') for p in subset],return_tensors='pt').to(device)
            f=model(**tensors).last_hidden_state[:,0].float()
            f=torch.nn.functional.normalize(f,dim=-1).cpu()
            features.update(zip(subset,f))
    del model,processor;gc.collect();torch.cuda.empty_cache()
    for r in rows:
        baseline=indexed[r['source_id'],r['method'],r['arm'],0.]
        r['csd_gain']=r['csd_target']-baseline['csd_target']
        image=np.asarray(Image.open(r['image_path']).convert('RGB'),dtype=np.float32)/255
        for name,path in [('input',r['source_path']),('recon',baseline['image_path'])]:
            reference=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
            assert image.shape==reference.shape==(512,512,3)
            r['dino_'+name]=float(features[r['image_path']] @ features[path])
            r['ssim_'+name]=float(calculate_ssim(reference,image))
            value=float(calculate_psnr(reference,image))
            # Mathematical +inf for identical images is recorded explicitly, never clipped.
            r['psnr_'+name]=value if math.isfinite(value) else None
            r['identical_'+name]=bool(np.array_equal(reference,image))
            r['mse_'+name]=float(np.mean((reference-image)**2))
            assert math.isfinite(r['dino_'+name]) and math.isfinite(r['ssim_'+name])
            assert r['psnr_'+name] is not None or r['identical_'+name]
        assert math.isfinite(r['csd_target']) and math.isfinite(r['csd_gain'])
    result=dict(rows=rows,n_images=expected,metrics={'csd_target':'mean cosine to32 independent Ukiyo-e references','csd_gain':'CSD minus same image/method unsteered reconstruction','dino':'normalized DINOv2-small CLS cosine','psnr':'RGB [0,1]; null iff identical (+infinity)','ssim':'mean of three channel skimage SSIM data_range1'},optional_metrics={'LPIPS':'not enabled; no assets acquired'},assets=json.loads((assets/'assets.json').read_text()))
    (run/'quality.json').write_text(json.dumps(result,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--assets',required=True);p.add_argument('--expected',type=int,required=True);a=p.parse_args();main(a.run,a.data,a.assets,a.expected)
