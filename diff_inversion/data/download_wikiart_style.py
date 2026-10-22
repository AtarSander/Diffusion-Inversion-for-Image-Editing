# ABOUTME: Streams huggan/wikiart and saves a target-style positive image set plus a
# ABOUTME: style-capped mixed negative set, with a steering sources.jsonl manifest.
"""Build a WikiArt concept dataset for image-derived steering directions.

Usage:
    uv run python -m diff_inversion.data.download_wikiart_style \
        --style Ukiyo_e --out_dir data/steering/wikiart_ukiyo_e
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import fire
from datasets import load_dataset
from dotenv import load_dotenv

DATASET = "huggan/wikiart"


def normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def main(
    style: str,
    out_dir: str,
    count: int = 16,
    min_side: int = 768,
    seed: int = 42,
    negative_cap: int = 2,
    shuffle_buffer: int = 2048,
) -> None:
    """Collect `count` positives of `style` and `count` mixed-style negatives.

    Args:
        style: WikiArt style name, case/separator-insensitive (e.g. "Ukiyo_e").
        out_dir: New directory for images, sources.jsonl and download_meta.json.
        count: Images per class; the steering fitter needs an equal even count.
        min_side: Reject images whose shorter side is below this many pixels,
            so SDXL 1024x1024 encoding never upscales heavily.
        seed: Shuffle seed for the streamed dataset.
        negative_cap: Maximum negatives drawn from any single other style, so the
            negative mean approximates "average painting" rather than one style.
        shuffle_buffer: Streaming shuffle buffer size.
    """
    load_dotenv()
    if count < 2 or count % 2:
        raise ValueError("count must be even and >= 2 for the balanced sign null")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    stream = load_dataset(DATASET, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    names = stream.features["style"].names
    lookup = {normalize(name): index for index, name in enumerate(names)}
    if normalize(style) not in lookup:
        raise ValueError(f"Unknown style {style!r}; available: {names}")
    target = lookup[normalize(style)]
    positives: list[dict[str, str]] = []
    negatives: list[dict[str, str]] = []
    negative_styles: Counter[str] = Counter()
    scanned = 0
    started = time.time()
    for row in stream:
        scanned += 1
        if len(positives) == count and len(negatives) == count:
            break
        image = row["image"]
        if min(image.size) < min_side:
            continue
        style_name = names[row["style"]]
        if row["style"] == target and len(positives) < count:
            bucket, rows = "positive", positives
        elif (
            row["style"] != target
            and len(negatives) < count
            and negative_styles[style_name] < negative_cap
        ):
            bucket, rows = "negative", negatives
        else:
            continue
        filename = f"{bucket}_{len(rows):03d}_{style_name}.jpg"
        image.convert("RGB").save(out / filename, quality=95)
        rows.append({"id": filename.removesuffix(".jpg"), "path": filename, "label": bucket})
        if bucket == "negative":
            negative_styles[style_name] += 1
        print(
            f"scanned {scanned}: positives {len(positives)}/{count},"
            f" negatives {len(negatives)}/{count} (+{bucket} {style_name})",
            flush=True,
        )
    if len(positives) < count or len(negatives) < count:
        raise RuntimeError(
            f"Stream exhausted after {scanned} rows:"
            f" {len(positives)} positives, {len(negatives)} negatives"
        )
    manifest = out / "sources.jsonl"
    lines = [json.dumps(row) for row in positives + negatives]
    manifest.write_text("\n".join(lines) + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {
        "dataset": DATASET,
        "style": names[target],
        "count_per_class": count,
        "min_side": min_side,
        "seed": seed,
        "negative_cap": negative_cap,
        "shuffle_buffer": shuffle_buffer,
        "rows_scanned": scanned,
        "negative_style_histogram": dict(sorted(negative_styles.items())),
        "git_sha": commit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    (out / "download_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {manifest} and download_meta.json")


if __name__ == "__main__":
    fire.Fire(main)
