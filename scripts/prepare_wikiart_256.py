"""Bounded, revision-pinned WikiArt subset acquisition; based on download_wikiart_style.
CPU-only PyArrow/Pillow. Reads label columns by HTTP range, then at most 4 GiB
of selected shard bytes. Never mirrors the dataset. All selection precedes edits.
"""
import argparse, collections, hashlib, io, json, random, struct, urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageOps, __version__ as pillow_version
REV='d559852d2b232e0fcf195e775866964f0564f2b5'
BASE=f'https://huggingface.co/datasets/huggan/wikiart/resolve/{REV}/'
LIMIT=4*1024**3
traffic=0

def fetch(url,limit,headers=None):
 global traffic
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=60) as r:
  b=r.read(limit+1)
  if len(b)>limit: raise RuntimeError('Response exceeds bounded request')
  traffic+=len(b)
  if traffic>LIMIT: raise RuntimeError('Acquisition byte cap exceeded')
  return b

class Remote(io.RawIOBase):
 def __init__(self,url,size): self.url=url;self.size=size;self.pos=0
 def readable(self): return True
 def seekable(self): return True
 def tell(self): return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,n if n>=0 else self.size)
  if n<=0:return b''
  b=fetch(self.url+f'?range_start={self.pos}&range_length={n}',n,{'Range':f'bytes={self.pos}-{self.pos+n-1}'})
  if len(b)!=n:raise RuntimeError('Short range response')
  self.pos+=len(b);return b

sha=lambda b:hashlib.sha256(b).hexdigest()
def main(out):
 root=Path(out);root.mkdir(parents=True,exist_ok=False)
 (root/'original').mkdir();(root/'rgb512').mkdir()
 metadata=json.loads(fetch(BASE+'dataset_infos.json',1000000))
 info=next(iter(metadata.values())); names=info['features']['style']['names'];assert names[26]=='Ukiyo_e'
 listing=json.loads(fetch(f'https://huggingface.co/api/datasets/huggan/wikiart/tree/{REV}/data?limit=100',1000000))
 shards=sorted(listing,key=lambda x:x['path']);random.Random(42).shuffle(shards)
 scanned=0;candidates=[]
 for shard in shards[:24]:
  f=pq.ParquetFile(Remote(BASE+shard['path'],shard['size']))
  labels=f.read(columns=['style']).column('style').to_pylist();scanned+=len(labels)
  assert scanned<=30000
  candidates.append((sum(x==26 for x in labels),shard,labels))
  print('metadata',shard['path'],len(labels),'target',candidates[-1][0],flush=True)
 # Prefer target-rich shards to bound transferred image bytes; deterministic ties.
 candidates.sort(key=lambda x:(-x[0],x[1]['path']))
 selected=[];seen_content=set();seen_processed=set();hist=collections.Counter();retained=0;exclusions=collections.Counter()
 for _,shard,labels in candidates:
  if sum(r['style_id']==26 for r in selected)>=96 and sum(r['style_id']!=26 for r in selected)>=160:break
  if traffic+shard['size']>LIMIT:break
  raw=fetch(BASE+shard['path']+'?download=true',shard['size'])
  pf=pq.ParquetFile(io.BytesIO(raw));idx=0
  for batch in pf.iter_batches(batch_size=16):
   for row in batch.to_pylist():
    rowidx=idx;idx+=1;s=row['style']
    target_count=sum(r['style_id']==26 for r in selected)
    other_count=len(selected)-target_count
    if (s==26 and target_count>=96) or (s!=26 and (other_count>=160 or hist[s]>=7)):continue
    b=row['image']['bytes'];im=ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert('RGB')
    if min(im.size)<512:exclusions['min_side_512']+=1;continue
    content=sha(struct.pack('>II',*im.size)+im.tobytes())
    p=ImageOps.fit(im,(512,512),method=Image.Resampling.LANCZOS,centering=(.5,.5));ph=sha(p.tobytes())
    if content in seen_content or ph in seen_processed:exclusions['duplicate']+=1;continue
    sid=Path(shard['path']).stem+f':{rowidx}'
    stem=sid.replace(':','_');op=root/'original'/f'{stem}.bin';pp=root/'rgb512'/f'{stem}.png'
    buf=io.BytesIO();p.save(buf,format='PNG',compress_level=9);pb=buf.getvalue()
    if retained+len(b)+len(pb)>1024**3:raise RuntimeError('Retained image cap')
    op.write_bytes(b);pp.write_bytes(pb);retained+=len(b)+len(pb)
    seen_content.add(content);seen_processed.add(ph);hist[s]+=1
    selected.append(dict(id=sid,source_id=sid,source_shard=shard['path'],source_row=rowidx,source_path_field=row['image'].get('path'),style_id=s,style_name=names[s],artist_id=row['artist'],genre_id=row['genre'],original_path=str(op.relative_to(root)),path=str(pp.relative_to(root)),original_size=list(im.size),original_sha256=sha(b),decoded_sha256=content,preprocessed_sha256=sha(pb),preprocessed_pixel_sha256=ph,original_bytes=len(b),preprocessed_bytes=len(pb)))
  print('images',len(selected),'target',sum(r['style_id']==26 for r in selected),'bytes',traffic,flush=True)
  del pf,raw
 target=[r for r in selected if r['style_id']==26];other=[r for r in selected if r['style_id']!=26]
 random.Random(42).shuffle(target)
 # Round-robin per-style negatives makes construction and edit distributions explicit.
 grouped={s:[r for r in other if r['style_id']==s] for s in sorted(hist) if s!=26}
 for s,rows in grouped.items():random.Random(42+s).shuffle(rows)
 other=[]
 while any(grouped.values()):
  for s in grouped:
   if grouped[s]:other.append(grouped[s].pop())
 splits={'construction':[(r,'positive') for r in target[:64]]+[(r,'negative') for r in other[:64]],'reference':[(r,'positive') for r in target[64:96]],'edit':[(r,'negative') for r in other[64:160]]}
 hashes={}
 for name,rows in splits.items():
  text=''.join(json.dumps(dict(r,label=label,split=name,selection_seed=42),sort_keys=True)+'\n' for r,label in rows)
  (root/f'{name}.jsonl').write_text(text);hashes[name]=sha(text.encode())
 report=dict(dataset='huggan/wikiart',revision=REV,style_names=names,seed=42,candidate_records=scanned,metadata_shards=[x[1]['path'] for x in candidates],selection='seed42 shard order; 24-shard bounded metadata pool; target-rich shard priority; max7 per non-target style; round-robin split',counts={k:len(v) for k,v in splits.items()},style_histograms={k:dict(collections.Counter(r['style_name'] for r,_ in v)) for k,v in splits.items()},exclusions=dict(exclusions),measured_response_bytes=traffic,retained_image_bytes=retained,unmeasured_network_overhead=True,preprocessing='EXIF transpose RGB, min side512, center crop ImageOps.fit512 LANCZOS, PNG compress9',pillow=pillow_version,pyarrow='23.0.1',manifest_sha256=hashes,complete=len(target)==96 and len(other)==160)
 (root/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['complete']:raise RuntimeError('Bounded acquisition incomplete; see acquisition.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
