"""Create a deterministic SDXL OOD prompt manifest from PIE-Bench source prompts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _path(value: str) -> Path:
    return Path(to_absolute_path(value)).resolve()


def _clean_prompt(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("[", "").replace("]", "").strip()


def prepare_prompts(mapping_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected object mapping in {mapping_file}, got {type(mapping)!r}")

    records: list[dict[str, Any]] = []
    rejected = Counter()
    for pie_bench_id, item in sorted(mapping.items()):
        if not isinstance(item, dict):
            rejected["non_object_record"] += 1
            continue
        prompt = _clean_prompt(item.get("original_prompt"))
        if not prompt:
            rejected["empty_original_prompt"] += 1
            continue
        record = {
            "prompt": prompt,
            "source_dataset": "pie_bench",
            "pie_bench_id": str(pie_bench_id),
            "editing_type_id": str(item.get("editing_type_id", "")),
        }
        records.append(record)

    summary = {
        "input_records": len(mapping),
        "selected_records": len(records),
        "rejected": dict(sorted(rejected.items())),
        "ordering": "lexicographic PIE-Bench mapping key",
        "prompt_field": "original_prompt with square brackets removed",
    }
    return records, summary


@hydra.main(config_path="../../config", config_name="prepare_pie_bench_ood_prompts", version_base=None)
def main(cfg: DictConfig) -> None:
    mapping_file = _path(str(cfg.data.mapping_file))
    output_path = _path(str(cfg.output_path))
    summary_path = _path(str(cfg.summary_path))
    records, summary = prepare_prompts(mapping_file)
    if len(records) != 700:
        raise ValueError(f"Expected all 700 PIE-Bench records, selected {len(records)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
