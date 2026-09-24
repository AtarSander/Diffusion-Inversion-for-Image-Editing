#!/usr/bin/env python3
"""Create deterministic PIE-Bench subsets used by reconstruction diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pixel-analysis", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-count", type=int, default=1)
    return parser.parse_args()


def cleaned_prompt(value: str) -> str:
    return value.replace("[", "").replace("]", "")


def write_dataset(root: Path, mapping: dict, source_images: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mapping_file.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    images_link = root / "annotation_images"
    if images_link.is_symlink():
        if images_link.resolve() != source_images.resolve():
            images_link.unlink()
        else:
            return
    if images_link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink {images_link}")
    images_link.symlink_to(source_images.resolve(), target_is_directory=True)


def main() -> None:
    args = parse_args()
    source_mapping = json.loads((args.data_root / "mapping_file.json").read_text())
    pixel_analysis = json.loads(args.pixel_analysis.read_text())

    noop = {}
    for key, item in source_mapping.items():
        record = dict(item)
        record["editing_prompt"] = cleaned_prompt(record["original_prompt"])
        record["editing_instruction"] = "No-op control: target prompt equals source prompt"
        record["blended_word"] = ""
        noop[key] = record

    worst_ids = {
        str(row["file_id"])
        for row in pixel_analysis["editors"]["pix2pix-zero"]["worst_normalized_mse"][:20]
    }
    worst = {key: item for key, item in source_mapping.items() if str(key) in worst_ids}
    if len(worst) != 20:
        raise ValueError(f"Expected exactly 20 Pix2Pix-Zero failures, found {len(worst)}")

    smoke_keys = sorted(source_mapping, key=lambda value: int(value))[: args.smoke_count]
    smoke = {key: source_mapping[key] for key in smoke_keys}
    smoke_noop = {key: noop[key] for key in smoke_keys}
    source_images = args.data_root / "annotation_images"

    write_dataset(args.output_root / "noop", noop, source_images)
    write_dataset(args.output_root / "pix2pix_worst20", worst, source_images)
    write_dataset(args.output_root / "smoke", smoke, source_images)
    write_dataset(args.output_root / "smoke_noop", smoke_noop, source_images)

    summary = {
        "source_count": len(source_mapping),
        "noop_count": len(noop),
        "pix2pix_worst20_ids": sorted(worst_ids, key=int),
        "smoke_ids": smoke_keys,
        "source_mapping": str((args.data_root / "mapping_file.json").resolve()),
        "pixel_analysis": str(args.pixel_analysis.resolve()),
    }
    (args.output_root / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
