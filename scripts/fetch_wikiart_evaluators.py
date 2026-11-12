"""Download only pinned public evaluator bytes; never deserialize model weights."""
import hashlib,json,urllib.request,argparse
from pathlib import Path
ASSETS=[('csd','tomg-group-umd/CSD-ViT-L','5bc26a6fb0487f3f00a2a7313135103a005b1b67',['pytorch_model.bin','config.json']),('dino','facebook/dinov2-small','ed25f3a31f01632728cabb09d1542f84ab7b0056',['model.safetensors','config.json','preprocessor_config.json']),('csd_processor','openai/clip-vit-large-patch14','32bd64288804d66eefd0ccbe215aa642df71cc41',['config.json','preprocessor_config.json'])]
def main(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);records=[];total=0
 for name,model,rev,files in ASSETS:
  dest=root/name;dest.mkdir()
  for file in files:
   url=f'https://huggingface.co/{model}/resolve/{rev}/{file}';h=hashlib.sha256();size=0
   with urllib.request.urlopen(url,timeout=60) as response,(dest/file).open('xb') as out:
    while b:=response.read(1024**2):
     total+=len(b);size+=len(b)
     if total>4*1024**3:raise RuntimeError('Evaluator download cap')
     out.write(b);h.update(b)
   records.append(dict(name=name,model=model,revision=rev,file=file,size=size,sha256=h.hexdigest()))
   (root/'assets.json').write_text(json.dumps(dict(files=records,total_bytes=total,complete=False),indent=2))
   print(name,file,size,h.hexdigest(),flush=True)
 (root/'assets.json').write_text(json.dumps(dict(files=records,total_bytes=total,complete=True),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
