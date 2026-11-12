"""Restricted metadata-only preflight for the pinned SD1.5 attention-only LoRA.

Never reads ZIP tensor storage members. No torch imports, pickle globals, models,
or tensor constructors are executed. All REDUCE globals map to local inert stubs.
Architecture enumeration is deliberately limited to standard SD1 UNet configs.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
CHECKPOINT_SHA = '173e03303012884c20f7c8084fe9a054efcdfb9a2a9f38aeeea52dd1d9eb6b3a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


class Storage:
    pass


def tensor_metadata(storage, offset, shape, stride, requires_grad, hooks):
    if not isinstance(storage, dict) or storage.get('kind') != 'storage':
        raise ValueError('Unsupported storage reference')
    if not all(type(n) is int and n >= 0 for n in shape):
        raise ValueError('Invalid shape')
    return dict(shape=list(shape), stride=list(stride), offset=offset, dtype='float32')


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): OrderedDict,
                   ('torch', 'FloatStorage'): Storage,
                   ('torch._utils', '_rebuild_tensor_v2'): tensor_metadata}
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported pickle global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != 'storage' or pid[1] is not Storage:
            raise ValueError('Unsupported persistent storage layout')
        return dict(kind='storage', key=pid[2], location=pid[3], elements=pid[4])


def metadata(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 4*1024*1024:
            raise ValueError('Unsupported checkpoint container')
        return MetadataOnly(io.BytesIO(z.read(names[0]))).load()


def expected_targets(c):
    # Reject other transformer layouts rather than extrapolating from a single key.
    assert c['_class_name'] == 'UNet2DConditionModel'
    assert c['layers_per_block'] == 2 and c['attention_head_dim'] == 8
    assert c['block_out_channels'] == [320,640,1280,1280]
    assert c['cross_attention_dim'] == 768
    assert c['down_block_types'] == ['CrossAttnDownBlock2D']*3+['DownBlock2D']
    assert c['up_block_types'] == ['UpBlock2D']+['CrossAttnUpBlock2D']*3
    assert c.get('transformer_layers_per_block',1) == 1
    assert c.get('mid_block_type','UNetMidBlock2DCrossAttn') == 'UNetMidBlock2DCrossAttn'
    assert not c.get('only_cross_attention',False) and not c.get('dual_cross_attention',False)
    assert not c.get('use_linear_projection',False) and not c.get('addition_embed_type')
    blocks = []
    for i,width in enumerate(c['block_out_channels'][:3]):
        blocks += [(f'down_blocks.{i}.attentions.{j}.transformer_blocks.0',width) for j in range(2)]
    blocks += [('mid_block.attentions.0.transformer_blocks.0',1280)]
    for i,width in enumerate(reversed(c['block_out_channels'])):
        if i: blocks += [(f'up_blocks.{i}.attentions.{j}.transformer_blocks.0',width) for j in range(3)]
    targets = {}
    for block,width in blocks:
        for attn in ('attn1','attn2'):
            for projection in ('to_q','to_k','to_v','to_out.0'):
                inp = 768 if attn == 'attn2' and projection in ('to_k','to_v') else width
                targets[f'{block}.{attn}.{projection}'] = [width,inp]
    return targets,[b for b,w in blocks if b.startswith('up_blocks.')]


def check(state,c):
    targets,selected = expected_targets(c)
    expected = {}
    for target,(out,inp) in targets.items():
        expected[target+'.lora_A.weight'] = [16,inp]
        expected[target+'.lora_B.weight'] = [out,16]
    missing = sorted(set(expected)-set(state)); unexpected = sorted(set(state)-set(expected))
    mismatched = [dict(key=k,expected=expected[k],observed=state[k].get('shape'))
                  for k in sorted(set(expected)&set(state)) if state[k].get('shape') != expected[k]]
    return dict(passed=not(missing or unexpected or mismatched),
        expected_targets=len(targets), checked_targets=sum(all(t+s in state for s in ('.lora_A.weight','.lora_B.weight')) for t in targets),
        expected_tensors=len(expected), checked_tensors=len(set(expected)&set(state)),
        missing_count=len(missing),unexpected_count=len(unexpected),mismatched_count=len(mismatched),
        missing=missing,unexpected=unexpected,mismatched=mismatched,rank=16,
        cross_attention_input=768,block_widths=c['block_out_channels'],selected_sa_layers=selected,
        tensors={k:dict(expected_shape=expected.get(k),**v) for k,v in state.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); c=json.loads(Path(a.config).read_text());state=metadata(a.checkpoint)
    result=check(state,c)
    result.update(model_id=MODEL,revision=REVISION,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=sha(a.checkpoint),
        architecture_sha256=sha(a.config),architecture_source=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/unet/config.json',
        mode='stdlib restricted metadata parser; ZIP data.pkl only; tensor storage never read except byte-only hash',
        limitations=['Architecture correspondence only; historical training loss and exact historical steering identity unproven.',
                    'Expected keys enumerated from standard SD1 UNet config; runtime full target check is also required before loading adapter.'])
    assert result['checkpoint_sha256'] == CHECKPOINT_SHA
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__':main()
