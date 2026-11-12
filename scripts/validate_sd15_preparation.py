"""Non-ML static/negative checks; optional byte validation of all prepared inputs."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import os
import json
from pathlib import Path
import pickle

repo=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preflight',repo/'scripts/preflight_sd15_adapter.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def main(verify_inputs=False):
    config=json.loads((Path(os.environ['VISION_SD15_EVIDENCE_DIR'])/'unet_config.json').read_text())
    evidence=json.loads((Path(os.environ['VISION_SD15_EVIDENCE_DIR'])/'preflight.json').read_text())
    state={k:{'shape':v['shape']} for k,v in evidence['tensors'].items()}
    assert p.check(state,config)['passed']
    key=next(k for k in state if 'attn2.to_k.lora_A' in k)
    bad=copy.deepcopy(state);bad[key]['shape']=[16,2048]
    assert p.check(bad,config)['mismatched_count']==1
    bad=copy.deepcopy(state);bad.pop(key)
    assert p.check(bad,config)['missing_count']==1
    bad=copy.deepcopy(state);bad['unsupported.lora_A.weight']={'shape':[16,768]}
    assert p.check(bad,config)['unexpected_count']==1
    badconfig=copy.deepcopy(config);badconfig['cross_attention_dim']=2048
    try:p.expected_targets(badconfig)
    except AssertionError:pass
    else:raise AssertionError('SDXL-like config accepted')
    # A pickle with a forbidden global must fail without invoking the global.
    try:p.MetadataOnly(io.BytesIO(b'cos\nsystem\n.')).load()
    except ValueError:pass
    else:raise AssertionError('Unsafe pickle global accepted')
    # Exercise conditioning dispatch without importing torch/diffusers or instantiating models.
    tree=ast.parse((repo/'diff_inversion/steering/images.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='conditioning')
    node.decorator_list=[];node.returns=None
    for a in node.args.args:a.annotation=None
    class FakeSD15:
        device='cpu'
        def encode_prompt(self,**kwargs):
            assert 'prompt_2' not in kwargs and 'negative_prompt_2' not in kwargs
            assert kwargs['do_classifier_free_guidance'] is True
            return 'positive','negative'
    ns={'StableDiffusionPipeline':FakeSD15}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'conditioning-test','exec'),ns)
    assert ns['conditioning'](FakeSD15(),'reader',512)=={'prompt_embeds':'positive','negative_prompt_embeds':'negative'}
    # Execute only the pure layer selector from its AST; no ML imports.
    tree=ast.parse((repo/'diff_inversion/steering/directions.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_layers')
    node.returns=None
    for a in node.args.args:a.annotation=None
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'layers-test','exec'),ns)
    selected=evidence['selected_sa_layers']
    targets,_=p.expected_targets(config)
    order=list(dict.fromkeys(k.rsplit('.attn',1)[0] for k in targets))
    assert ns['select_layers'](order,'up_blocks.1,up_blocks.2,up_blocks.3')==selected and len(selected)==9
    empty=json.loads((repo/'config/steering/outline_reader_empty.json').read_text())
    different=json.loads((repo/'config/steering/outline_reader_different.json').read_text())
    assert different.pop('reader_prompts_by_label')=={'positive':'a large outline triangle on a white background','negative':'a small outline triangle on a white background'}
    assert empty==different and empty['inversion_prompt']=='' and empty['size']==512
    assert empty['model']['id']==p.MODEL and empty['model']['revision']==p.REVISION
    changed=['diff_inversion/steering/images.py','diff_inversion/steering/runtime.py','diff_inversion/steering/lora.py',
        'scripts/run_outline_reader_ablation.py','scripts/preflight_sd15_adapter.py','scripts/prepare_sd15_inputs.py',
        'scripts/diagnose_outline_readers.py','scripts/validate_sd15_preparation.py']
    for name in changed:ast.parse((repo/name).read_text())
    if verify_inputs:
        root=repo/'data/steering/sd15_verified'
        saved=json.loads((Path(os.environ['VISION_SD15_EVIDENCE_DIR'])/'input_evidence.json').read_text())
        for rel,sha in saved['normalized_sha256'].items():
            assert hashlib.sha256((root/'inputs512'/rel).read_bytes()).hexdigest()==sha,rel
        for family,n in [('triangles64_outline',128),('circles8_eval',8)]:
            manifest=root/'inputs512'/family/'sources.jsonl'
            rows=[json.loads(s) for s in manifest.read_text().splitlines()]
            assert len(rows)==n and len({r['id'] for r in rows})==n
            for r in rows:assert (manifest.parent/r['path']).is_file()
    print(json.dumps(dict(status='passed',complete_metadata_match=True,negative_shape_key_layout_checks=True,
        forbidden_global_rejected=True,sd15_conditioning_dispatch=True,nine_valid_up_layers=True,
        configs_only_reader_prompts_differ=True,ast_files=len(changed),input_hashes_checked=verify_inputs)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-inputs',action='store_true');a=ap.parse_args();main(a.verify_inputs)
