"""Prepare a lightweight SD1.5 trajectory view for paired inversion evaluation.

The manifest follows the 1,522-prompt SDXL inversion-quality ordering but maps
each prompt to its corresponding SD1.5 reference trajectory. The SD1.5 models
were trained on these prompts, so this is intentionally labelled an in-sample
diagnostic rather than a held-out result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


INPUT_NAMES = (
    "conditioning.pt",
    "final.png",
    "prompt.json",
    "timesteps.json",
    "latents",
    "targets",
)


def _sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("sample_*") if path.is_dir())


def _read_prompt(sample_dir: Path) -> str:
    with (sample_dir / "prompt.json").open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Missing prompt in {sample_dir / 'prompt.json'}")
    return prompt


def create_manifest(
    source_root: Path,
    source_prompts_jsonl: Path | None,
    prompt_root: Path,
    output_path: Path,
    expected_samples: int,
) -> dict[str, object]:
    source_by_prompt: dict[str, Path] = {}
    duplicates: set[str] = set()
    if source_prompts_jsonl is not None:
        source_records = [
            json.loads(line)
            for line in source_prompts_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_items = (
            (source_root / f"sample_{index:06d}", str(record["prompt"]))
            for index, record in enumerate(source_records)
        )
    else:
        source_items = (
            (sample_dir, _read_prompt(sample_dir))
            for sample_dir in _sample_dirs(source_root)
        )

    for sample_dir, prompt in source_items:
        if prompt in source_by_prompt:
            duplicates.add(prompt)
        else:
            source_by_prompt[prompt] = sample_dir
    if duplicates:
        raise ValueError(f"Found {len(duplicates)} duplicate prompts in {source_root}")

    prompt_samples = _sample_dirs(prompt_root)
    if len(prompt_samples) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} prompt samples in {prompt_root}, "
            f"found {len(prompt_samples)}"
        )

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for eval_index, prompt_sample in enumerate(prompt_samples):
        prompt = _read_prompt(prompt_sample)
        source_sample = source_by_prompt.get(prompt)
        if source_sample is None:
            missing.append(prompt)
            continue
        rows.append(
            {
                "eval_index": eval_index,
                "eval_sample": f"sample_{eval_index:06d}",
                "source_sample": source_sample.name,
                "prompt_source_sample": prompt_sample.name,
                "prompt": prompt,
            }
        )
    if missing:
        raise ValueError(f"Missing {len(missing)} prompts from {source_root}")

    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return {
        "num_samples": len(rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_root": source_root.as_posix(),
        "prompt_root": prompt_root.as_posix(),
        "protocol": "sd15_in_sample_on_sdxl_heldout_prompt_ids",
    }


def build_view(
    source_root: Path,
    manifest_path: Path,
    output_root: Path,
    max_samples: int | None = None,
) -> int:
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_root / "run_config.json", output_root / "run_config.json")

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_samples is not None:
        rows = rows[:max_samples]
    for row in rows:
        source_sample = source_root / str(row["source_sample"])
        view_sample = output_root / str(row["eval_sample"])
        view_sample.mkdir()
        for name in INPUT_NAMES:
            source_path = source_sample / name
            if source_path.exists():
                os.symlink(source_path.resolve(), view_sample / name)

        meta_path = source_sample / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(
            {
                "inversion_quality_source_sample": source_sample.name,
                "inversion_quality_protocol": "sd15_in_sample_on_sdxl_heldout_prompt_ids",
            }
        )
        (view_sample / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-prompts-jsonl", type=Path)
    parser.add_argument("--prompt-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=1522)
    parser.add_argument("--view-root", type=Path)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    if args.prompt_root is not None:
        summary = create_manifest(
            source_root=source_root,
            source_prompts_jsonl=(
                args.source_prompts_jsonl.resolve()
                if args.source_prompts_jsonl is not None
                else None
            ),
            prompt_root=args.prompt_root.resolve(),
            output_path=manifest_path,
            expected_samples=args.expected_samples,
        )
        summary_path = manifest_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    if args.view_root is not None:
        count = build_view(
            source_root,
            manifest_path,
            args.view_root.resolve(),
            max_samples=args.max_samples,
        )
        if count != args.expected_samples:
            raise ValueError(f"Expected {args.expected_samples} manifest rows, found {count}")
        print(f"Prepared {count} samples in {args.view_root}")


if __name__ == "__main__":
    main()
