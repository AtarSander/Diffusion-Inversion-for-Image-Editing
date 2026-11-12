"""Download the official archive; retain only unmodified held-out image pairs."""
import argparse
import io
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from PIL import Image
import yaml
from prepare_bbbc021 import download, sha256, atomic_json, write_csv


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='config/data/edges2shoes_prepare.yaml')
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
    root, out = Path(cfg['raw_dir']), Path(cfg['manifest_dir'])
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='edges2shoes-') as temp:
        archive = Path(temp) / 'edges2shoes.tar.gz'
        download(cfg['url'], archive)
        digest = sha256(archive)
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if len(parts) != 3 or parts[:2] != ('edges2shoes', cfg['split']):
                    continue
                if not member.isfile() or not parts[2].lower().endswith(('.jpg', '.png')):
                    continue
                data = tar.extractfile(member).read()
                with Image.open(io.BytesIO(data)) as im:
                    assert im.mode == 'RGB' and im.size == (512, 256), member.name
                    im.verify()
                target = root / cfg['split'] / parts[2]
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + '.tmp')
                tmp.write_bytes(data)
                tmp.replace(target)
                rows.append(dict(image_path=str(target.relative_to(root)), image_id=target.stem,
                                 split=cfg['split'], class_idx=0, sha256=sha256(target)))
    rows.sort(key=lambda r: r['image_path'])
    assert len(rows) == cfg['expected_images'], len(rows)
    assert len({r['image_path'] for r in rows}) == len(rows)
    write_csv(out / 'val.csv', rows)
    atomic_json(out / 'summary.json', dict(status='complete', images=len(rows),
        source=cfg['url'], archive_sha256=digest, manifest_sha256=sha256(out / 'val.csv'),
        protocol='official validation pairs, left=edge/right=shoe, no pixel changes',
        class_labels='not supplied; class_idx=0 placeholder, kNN not reported'))
    print(f'COMPLETE: {len(rows)} official validation pairs; no training images extracted', flush=True)


if __name__ == '__main__':
    main()
