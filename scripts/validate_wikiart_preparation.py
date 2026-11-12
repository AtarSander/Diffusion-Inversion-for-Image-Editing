"""Non-ML, read-only checks of frozen images/configs and paired aggregation."""
import ast,hashlib,importlib.util,json,struct,sys
from pathlib import Path
from PIL import Image,ImageOps

def check(data,repo):
    data,repo=Path(data),Path(repo);meta=json.loads((data/'acquisition.json').read_text())
    allrows=[];counts={};hashes={}
    for split,n in [('construction',128),('reference',32),('edit',96)]:
        b=(data/f'{split}.jsonl').read_bytes();assert hashlib.sha256(b).hexdigest()==meta['manifest_sha256'][split]
        rows=[json.loads(l) for l in b.splitlines()];assert len(rows)==n;counts[split]=n;allrows+=rows
        assert all(r['split']==split for r in rows)
    assert len({r['id'] for r in allrows})==256
    assert len({r['decoded_sha256'] for r in allrows})==256
    assert len({r['preprocessed_pixel_sha256'] for r in allrows})==256
    assert sum(r['style_id']==26 for r in allrows)==96
    assert sum(r['label']=='positive' for r in allrows if r['split']=='construction')==64
    assert all(r['style_id']==26 for r in allrows if r['split']=='reference')
    assert all(r['style_id']!=26 for r in allrows if r['split']=='edit')
    for r in allrows:
        original=data/r['original_path'];processed=data/r['path']
        assert hashlib.sha256(original.read_bytes()).hexdigest()==r['original_sha256']
        assert hashlib.sha256(processed.read_bytes()).hexdigest()==r['preprocessed_sha256']
        im=ImageOps.exif_transpose(Image.open(original)).convert('RGB');assert min(im.size)>=512
        assert hashlib.sha256(struct.pack('>II',*im.size)+im.tobytes()).hexdigest()==r['decoded_sha256']
        regenerated=ImageOps.fit(im,(512,512),method=Image.Resampling.LANCZOS,centering=(.5,.5))
        assert regenerated.tobytes()==Image.open(processed).convert('RGB').tobytes()
        hashes[str(processed)]=r['preprocessed_sha256']
    configs=[json.loads((repo/f'config/steering/wikiart_reader_{a}.json').read_text()) for a in ('empty','different')]
    maps=[c.pop('reader_prompts_by_source_id') for c in configs];assert configs[0]==configs[1]
    ids={r['id'] for r in allrows if r['split']=='construction'};assert set(maps[0])==set(maps[1])==ids
    assert all(v=='' for v in maps[0].values())
    for r in allrows:
        if r['split']=='construction':
            style='ukiyo-e' if r['style_id']==26 else r['style_name'].replace('_',' ').lower()
            assert maps[1][r['id']]==f'a painting in the {style} style'
    assert configs[0]['inversion_prompt']=='' and configs[0]['steps']==50 and configs[0]['size']==512
    tree=ast.parse((repo/'diff_inversion/steering/config.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_reader_prompts','resolve_reader_prompts')]
    ns={'Config':dict,'Source':dict}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<reader-validation>','exec'),ns)
    sources=[{'id':'a','label':'positive'},{'id':'b','label':'negative'},{'id':'c','label':'negative'}]
    c={'reader_prompt':'','inversion_prompt':'','reader_prompts_by_source_id':{'a':'ukiyo-e','b':'cubism','c':'baroque'}}
    assert ns['resolve_reader_prompts'](c,sources)==c['reader_prompts_by_source_id']
    assert c['inversion_prompt']==''
    for broken in [{'a':'ukiyo-e'},{'a':'ukiyo-e','b':'cubism','c':'baroque','d':'extra'}]:
        try:ns['resolve_reader_prompts'](dict(c,reader_prompts_by_source_id=broken),sources)
        except ValueError:pass
        else:raise AssertionError('Reader coverage validation failed')
    assert ns['resolve_reader_prompts']({'reader_prompt':''},sources)==dict.fromkeys(['a','b','c'],'')
    spec=importlib.util.spec_from_file_location('summary',repo/'scripts/summarize_wikiart.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    metrics=['csd_target','csd_gain','dino_input','dino_recon','ssim_input','ssim_recon','mse_input','mse_recon','psnr_input','psnr_recon']
    for i in range(4):
        for method in ('DDIM','MW'):
            for arm in ('empty','different'):
                for strength in (0.,1.,2.):
                    values={m:float(i)+strength*(2 if arm=='different' else 1) for m in metrics}
                    rows.append(dict(source_id=str(i),method=method,arm=arm,strength=strength,**values))
    result=module.summarize(rows,4)
    for r in result['rows']:
        assert r['paired_difference']['mean']==r['strength']
        assert r['paired_difference']['sem']==0
        assert r['paired_difference']['ci95']==[r['strength'],r['strength']]
    try:module.summarize(rows[:-1],4)
    except AssertionError:pass
    else:raise AssertionError('Missing-grid check failed')
    for name in ['prepare_wikiart_256','fetch_wikiart_evaluators','score_wikiart_edits','summarize_wikiart','run_wikiart_readers','wikiart_vector_stats']:
        ast.parse((repo/f'scripts/{name}.py').read_text())
    print(json.dumps(dict(status='passed',counts=counts,unique_ids=256,unique_content_hashes=256,decoded_images_checked=256,deterministic_preprocessing_rechecks=256,reader_only_config_difference=True,paired_bootstrap_fixture=True,missing_grid_rejected=True),indent=2))
if __name__=='__main__':check(*sys.argv[1:])
