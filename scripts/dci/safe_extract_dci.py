"""Inspect a checksum-verified DCI tar and extract only safe regular files/directories."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import subprocess
import getpass

state = Path(os.environ['DCI_ACQUISITION'])
base = Path(os.environ['DCI_PROJECT_DATA'])/'raw/DCI'
receipt = json.loads((state/'annotation_download.json').read_text())
if not receipt.get('verified'):
    raise RuntimeError('Require a successful download checksum receipt')
archive = Path(receipt['archive_path'])
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        h.update(block)
if h.hexdigest() != receipt['expected_sha256']:
    raise RuntimeError('Archive changed since download verification')
partial = base/'extracted.part'
final = base/'dataset'
if partial.exists() or final.exists():
    raise FileExistsError('Refusing to overwrite an existing extraction')
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    if len(members) > 100000:
        raise RuntimeError('Archive entry safety cap exceeded')
    total = sum(m.size for m in members)
    stats = os.statvfs(base)
    if total > 30*1024**3 or total*2 > stats.f_bavail*stats.f_frsize:
        raise RuntimeError('Archive size/headroom safety check failed')
    if len(members)*2 > stats.f_favail:
        raise RuntimeError('Insufficient inode headroom')
    quota_mount = os.environ['DCI_QUOTA_MOUNT']
    quota_text = subprocess.check_output(
        ['lfs','quota','-q','-u',getpass.getuser(),quota_mount], text=True, timeout=30)
    rows = [line.split() for line in quota_text.splitlines() if line.split() and line.split()[0] == quota_mount]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise RuntimeError('Cannot verify per-user inode quota; refusing extraction')
    quota_used = int(rows[0][5].rstrip('*'))
    quota_limit = int(rows[0][7].rstrip('*'))
    if quota_limit and quota_limit-quota_used < len(members)+256:
        raise RuntimeError('Insufficient per-user inode quota for extraction plus headroom')
    names = set()
    inventory = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or '\\' in member.name:
            raise RuntimeError('Unsafe archive pathname '+member.name)
        if not (member.isfile() or member.isdir()):
            raise RuntimeError('Archive links/special files are disallowed: '+member.name)
        normalized = str(path)
        if normalized in names:
            raise RuntimeError('Duplicate archive path '+normalized)
        names.add(normalized)
        inventory.append({'path':member.name,'bytes':member.size,'type':'file' if member.isfile() else 'directory'})
    (state/'archive_members.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'validated_members':len(members),'uncompressed_bytes':total,'first_paths':[x['path'] for x in inventory[:12]]}),flush=True)
    partial.mkdir()
    for member in members:
        target = partial.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            target.mkdir(parents=True,exist_ok=True)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            target.chmod(0o644)
partial.rename(final)
record={'archive_sha256':h.hexdigest(),'archive_members':len(members),'uncompressed_bytes':total,'extraction_root':str(final),'safe_paths_verified':True,'atomic_completion':True,'links_and_special_files':0}
(state/'extraction.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
