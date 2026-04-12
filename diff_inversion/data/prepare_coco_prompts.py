"""Prepare shuffled COCO prompt splits from raw caption annotation JSON."""

import argparse
import json
from pathlib import Path
import random
from typing import Any

from diff_inversion.data.coco import (
    get_coco_caption_source,
    get_processed_coco_dir,
    get_raw_coco_dir,
)


def normalize_prompt(text: str) -> str:
    """Collapse repeated whitespace and trim the prompt text."""
    return " ".join(text.strip().split())


def load_coco_captions(captions_json_path: Path) -> list[dict[str, Any]]:
    """Load caption annotations and convert them into prompt records."""
    with captions_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError("Input JSON does not contain 'annotations' field.")

    records: list[dict[str, Any]] = []
    for ann in data["annotations"]:
        caption = ann.get("caption")
        image_id = ann.get("image_id")
        annotation_id = ann.get("id")

        if caption is None:
            continue

        records.append(
            {
                "prompt": normalize_prompt(str(caption)),
                "image_id": image_id,
                "annotation_id": annotation_id,
            }
        )

    return records


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate prompts while keeping the first occurrence of each one."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for record in records:
        prompt = record["prompt"]
        if prompt in seen:
            continue
        seen.add(prompt)
        deduped.append(record)

    return deduped


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split prompt records into train, validation, and test partitions."""
    total = len(records)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    return train_records, val_records, test_records


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entrypoint for transforming raw COCO captions into prompt datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2014", choices=["2014", "2017"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    _, _, extracted_name = get_coco_caption_source(args.year, args.split)
    default_input_json = get_raw_coco_dir() / args.year / args.split / extracted_name
    default_output_dir = get_processed_coco_dir() / args.year / args.split

    input_json_path = Path(args.input_json) if args.input_json else default_input_json
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_json_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_json_path}\n"
            "Run the download step first or pass --input_json."
        )

    records = load_coco_captions(input_json_path)
    print(f"Loaded {len(records)} caption records from {input_json_path.name}")

    if args.deduplicate:
        before = len(records)
        records = deduplicate_records(records)
        after = len(records)
        print(f"Deduplicated prompts: {before} -> {after}")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    train_records, val_records, test_records = split_records(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    save_jsonl(records, output_dir / "all_prompts.jsonl")
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(val_records, output_dir / "val.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")

    summary = {
        "year": args.year,
        "split_source": args.split,
        "input_json_path": str(input_json_path),
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "deduplicate": args.deduplicate,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Saved prompts: {output_dir / 'all_prompts.jsonl'}")
    print(f"Saved prompts: {output_dir / 'train.jsonl'}")
    print(f"Saved prompts: {output_dir / 'val.jsonl'}")
    print(f"Saved prompts: {output_dir / 'test.jsonl'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
