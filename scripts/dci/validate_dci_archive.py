"""Validate official annotation JSON/splits in the complete archive without model imports."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from collections import Counter

base = Path(os.environ['DCI_PROJECT_DATA'])/'raw/DCI'
state = Path(os.environ['DCI_ACQUISITION'])
receipt = json.loads((state/'annotation_download.json').read_text())
if not receipt.get('verified'):
    raise RuntimeError('Require checksum-verified archive')
archive_hash = hashlib.sha256()
with Path(receipt['archive_path']).open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        archive_hash.update(block)
if archive_hash.hexdigest() != receipt['expected_sha256']:
    raise RuntimeError('Archive changed since download verification')
metadata = base/'metadata'
metadata.mkdir(exist_ok=True)
photos = base/'photos'
photos.mkdir(exist_ok=True)
annotations = {}
complete = {}
counts = Counter()
errors = []
image_members = []
split_bytes = None
catalog_tmp = metadata/'caption_catalog.jsonl.part'
with catalog_tmp.open('x') as catalog, tarfile.open(receipt['archive_path'], 'r|gz') as tar:
    for member in tar:
        if not member.isfile():
            continue
        p = PurePosixPath(member.name)
        if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp']:
            image_members.append(member.name)
        if p.suffix != '.json':
            continue
        body = tar.extractfile(member).read()
        try:
            data = json.loads(body)
        except Exception as exc:
            errors.append({'member':member.name,'error':str(exc)})
            continue
        counts['json_parsed'] += 1
        if p.name == 'splits.json':
            split_bytes = body
            continue
        group = p.parent.name
        if group not in ['annotations','complete']:
            counts['other_json'] += 1
            continue
        image = data['image']
        if PurePosixPath(image).is_absolute() or '..' in PurePosixPath(image).parts:
            raise ValueError('Unsafe referenced image path')
        target = annotations if group == 'annotations' else complete
        if p.name in target:
            raise ValueError('Duplicate annotation identity '+p.name)
        target[p.name] = {'member':member.name,'image':image,'width':data.get('width'),
                          'height':data.get('height'),'sha256':hashlib.sha256(body).hexdigest()}
        counts[group] += 1
        for field in ['short_caption','extra_caption','summaries','negatives']:
            if data.get(field):
                counts[group+'_'+field] += 1
        # Preserve all original text fields; do not choose a reconstruction prompt.
        caption = {'annotation_id':p.name,'representation':group,'image':image,
                   'source_archive_member':member.name,
                   'short_caption':data.get('short_caption'),'extra_caption':data.get('extra_caption'),
                   'mask_captions':{k:{field:v[field] for field in ['label','caption','parent','requirements','mask_quality'] if field in v}
                                    for k,v in data.get('mask_data',{}).items()},
                   'summaries':data.get('summaries'),'negatives':data.get('negatives')}
        catalog.write(json.dumps(caption,ensure_ascii=False,separators=(',',':'))+'\n')
if split_bytes is None:
    raise RuntimeError('Official splits.json absent')
splits = json.loads(split_bytes)
(metadata/'splits.json').write_bytes(split_bytes)
catalog_tmp.rename(metadata/'caption_catalog.jsonl')
by_id = {}
split_results = {}
for name,ids in splits.items():
    if not isinstance(ids,list):
        raise TypeError('Unknown split structure')
    missing = [i for i in ids if i not in annotations]
    duplicate_count = len(ids)-len(set(ids))
    split_results[name] = {'count':len(ids),'unique':len(set(ids)),'missing_annotations':missing,
                           'duplicate_ids':duplicate_count,'first_ids':ids[:5]}
    for ident in ids:
        if ident in by_id:
            raise ValueError('Overlapping official splits')
        by_id[ident]=name
images = set(x['image'] for x in annotations.values())
image_metadata_conflicts = [k for k in annotations if k in complete and
                            any(annotations[k][f] != complete[k][f] for f in ['image','width','height'])]
present = []
missing = []
corrupt = []
readable = []
for name in sorted(images):
    target = photos/name
    if target.is_file(): present.append(name)
    else: missing.append(name)
if present:
    from PIL import Image
    for name in present:
        try:
            with Image.open(photos/name) as im:
                size=im.size;im.verify()
            with Image.open(photos/name) as im:im.load()
            readable.append({'image':name,'dimensions':list(size)})
        except Exception as exc:corrupt.append({'image':name,'error':str(exc)})
with (metadata/'annotation_index.jsonl').open('x') as stream:
    for ident,row in sorted(annotations.items()):
        stream.write(json.dumps({'id':ident,'official_split':by_id.get(ident),'annotation':row,
                                'complete':complete.get(ident),'expected_photo_path':str(photos/row['image']),
                                'photo_present':(photos/row['image']).is_file()},ensure_ascii=False)+'\n')
(metadata/'missing_images.json').write_text(json.dumps(missing,indent=2)+'\n')
report={'archive_path':receipt['archive_path'],'archive_sha256':receipt['actual_sha256'],
        'counts':dict(counts),'json_errors':errors,'unique_referenced_images':len(images),
        'official_splits':split_results,'split_file':str(metadata/'splits.json'),
        'split_sha256':hashlib.sha256(split_bytes).hexdigest(),
        'annotations_not_in_splits':sorted(set(annotations)-set(by_id)),
        'complete_missing_annotations':sorted(set(annotations)-set(complete)),
        'image_metadata_conflicts':image_metadata_conflicts,'archive_image_files':image_members,
        'photos_path':str(photos),'available_unique_images':len(present),'missing_unique_images':len(missing),
        'images_checked':len(present),'readable_images':readable,'corrupt_present_images':corrupt,
        'image_validation_status':'no raw images available' if not present else 'checked',
        'annotation_status':'verified archive; extraction exists' if (base/'dataset').is_dir() else 'verified archive; directory extraction not present',
        'ready_for_reconstruction':len(missing)==0 and not corrupt and not errors,
        'caption_catalog':str(metadata/'caption_catalog.jsonl'),'index':str(metadata/'annotation_index.jsonl')}
(state/'dci_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
