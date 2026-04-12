"""Download and extract COCO caption annotations into the project's raw data directory."""

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlretrieve
import zipfile

from diff_inversion.data.coco import get_coco_caption_source, get_raw_coco_dir


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: Path) -> None:
    """Download a remote file to the requested path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    print(f"Saving to:   {output_path}")
    urlretrieve(url, output_path)
    print("Download finished.")


def extract_single_file(zip_path: Path, member_name: str, output_path: Path) -> None:
    """Extract a single member from a ZIP archive without unpacking the whole archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if member_name not in members:
            raise FileNotFoundError(
                f"{member_name} not found in archive.\n"
                f"Available members include e.g.: {members[:10]}"
            )

        with zf.open(member_name) as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"Extracted: {member_name} -> {output_path}")


def main() -> None:
    """CLI entrypoint for fetching COCO caption JSON files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2014", choices=["2014", "2017"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--keep_zip", action="store_true")
    args = parser.parse_args()

    url, member_name, extracted_name = get_coco_caption_source(args.year, args.split)
    output_dir = (
        Path(args.output_dir) if args.output_dir else get_raw_coco_dir() / args.year / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / Path(url).name
    captions_json_path = output_dir / extracted_name

    if not zip_path.exists():
        download_file(url, zip_path)
    else:
        print(f"ZIP already exists: {zip_path}")

    print(f"ZIP SHA256: {sha256_of_file(zip_path)}")

    if not captions_json_path.exists():
        extract_single_file(zip_path, member_name, captions_json_path)
    else:
        print(f"JSON already exists: {captions_json_path}")

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"Removed ZIP: {zip_path}")

    print("Done.")
    print(f"Saved raw JSON: {captions_json_path}")


if __name__ == "__main__":
    main()
