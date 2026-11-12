"""Download official BBBC021 and prepare a reproducible transfer-evaluation pool.

Archives are staged one at a time in TMPDIR, never extracted onto shared storage.
This is an explicit local protocol, not a reconstruction of OVG's missing split.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urljoin
import zipfile

import numpy as np
from PIL import Image
import yaml


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_name(path.name + '.partial')
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '5',
                    '--retry-delay', '5', '--connect-timeout', '60',
                    '--max-time', '7200', '--output', str(partial), url], check=True)
    partial.replace(path)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def channel_uint8(array, percentiles):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Expected finite single-channel TIFF')
    lo, hi = np.percentile(array, percentiles)
    if hi <= lo:
        raise ValueError('Degenerate channel intensity range')
    return np.rint(np.clip((array - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='config/data/bbbc021_prepare.yaml')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw, out = Path(cfg['raw_dir']), Path(cfg['manifest_dir'])
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(cfg, protocol='bbbc021_transfer_v1', paper_split=False)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; choose new output directories')
    atomic_json(protocol_path, protocol)
    page = raw / 'source_page.html'
    download(cfg['source_page'], page)
    archives = sorted(set(re.findall(
        r'https://data\.broadinstitute\.org/bbbc/BBBC021/BBBC021_v1_images_[^"\s<>]+\.zip',
        page.read_text())))
    if len(archives) != cfg['expected_archives']:
        raise ValueError(f'Expected {cfg["expected_archives"]} archives, found {len(archives)}')
    for name in ['image', 'moa', 'compound']:
        filename = f'BBBC021_v1_{name}.csv'
        download(urljoin(cfg['source_base'], filename), raw / filename)
    rows = list(csv.DictReader((raw / 'BBBC021_v1_image.csv').open()))
    moas = list(csv.DictReader((raw / 'BBBC021_v1_moa.csv').open()))
    labels = sorted({r['moa'] for r in moas})
    mapping = {(r['compound'], Decimal(r['concentration'])): r['moa'] for r in moas}
    atomic_json(out / 'classes.json', dict(enumerate(labels)))
    if len(rows) != cfg['expected_fields']:
        raise ValueError(f'Unexpected field count: {len(rows)}')
    atomic_json(out / 'downloads.json', {'archives': archives, 'metadata_sha256': {
        p.name: sha256(p) for p in sorted(raw.glob('*.csv'))}})
    print(f'Metadata ready: {len(rows)} fields, {len(archives)} archives, {len(labels)} MOA classes', flush=True)
    if args.metadata_only:
        return
    by_plate = {}
    for row in rows:
        by_plate.setdefault(row['Image_Metadata_Plate_DAPI'], []).append(row)
    manifests = []
    for number, url in enumerate(archives, 1):
        filename = url.rsplit('/', 1)[-1]
        plate = filename.removeprefix('BBBC021_v1_images_').removesuffix('.zip')
        records = sorted(by_plate.pop(plate), key=lambda r: (int(r['TableNumber']), int(r['ImageNumber'])))
        directory = raw / 'images' / plate
        directory.mkdir(parents=True, exist_ok=True)
        done = out / f'{plate}.json'
        if done.exists():
            saved = json.loads(done.read_text())
            for row in saved['rows']:
                if sha256(raw / row['path']) != row['sha256']:
                    raise ValueError(f'Corrupt prepared image: {row["path"]}')
            manifests.extend(saved['rows'])
            print(f'[{number}/{len(archives)}] verified {plate}', flush=True)
            continue
        print(f'[{number}/{len(archives)}] downloading and preparing {plate}', flush=True)
        with tempfile.TemporaryDirectory(prefix='bbbc021-') as temp:
            archive = Path(temp) / filename
            download(url, archive)
            archive_hash = sha256(archive)
            prepared = []
            with zipfile.ZipFile(archive) as z:
                members = {}
                for member in z.namelist():
                    if member.lower().endswith(('.tif', '.tiff')) and not member.startswith('__MACOSX/'):
                        basename = Path(member).name
                        if basename in members:
                            raise ValueError(f'Duplicate TIFF filename: {basename}')
                        members[basename] = member
                for row in records:
                    channels = []
                    for channel in cfg['channels_rgb']:
                        member = members[row[f'Image_FileName_{channel}']]
                        with Image.open(io.BytesIO(z.read(member))) as image:
                            channels.append(channel_uint8(np.asarray(image), cfg['percentiles']))
                    rgb = Image.fromarray(np.stack(channels, axis=-1))
                    rgb = rgb.resize((cfg['resolution'], cfg['resolution']), Image.Resampling.BILINEAR)
                    target = directory / f'{row["TableNumber"]}_{row["ImageNumber"]}.png'
                    temporary = target.with_suffix('.png.tmp')
                    rgb.save(temporary, format='PNG')
                    temporary.replace(target)
                    moa = mapping.get((row['Image_Metadata_Compound'], Decimal(row['Image_Metadata_Concentration'])), '')
                    prepared.append(dict(path=str(target.relative_to(raw)),
                        class_idx=labels.index(moa) if moa else -1, res_class=1,
                        moa=moa, plate=plate, well=row['Image_Metadata_Well_DAPI'],
                        compound=row['Image_Metadata_Compound'], concentration=row['Image_Metadata_Concentration'],
                        table=row['TableNumber'], image_number=row['ImageNumber'],
                        sha256=sha256(target)))
            atomic_json(done, dict(url=url, archive_sha256=archive_hash, rows=prepared))
            manifests.extend(prepared)
        print(f'[{number}/{len(archives)}] complete: {len(prepared)} RGB images', flush=True)
    if by_plate or len(manifests) != cfg['expected_fields']:
        raise ValueError('Archive/metadata coverage mismatch')
    manifests.sort(key=lambda r: (r['plate'], int(r['table']), int(r['image_number'])))
    labelled = [r for r in manifests if r['class_idx'] >= 0]
    write_csv(out / 'all.csv', manifests)
    write_csv(out / 'test.csv', labelled)
    atomic_json(out / 'summary.json', dict(status='complete', fields=len(manifests),
        evaluation_fields=len(labelled), unlabelled_fields=len(manifests)-len(labelled),
        archives=len(archives), classes=len(labels), protocol=protocol,
        test_sha256=sha256(out / 'test.csv'), all_sha256=sha256(out / 'all.csv')))
    print(f'COMPLETE: {len(manifests)} prepared images; {len(labelled)} labelled evaluation images', flush=True)


if __name__ == '__main__':
    main()
