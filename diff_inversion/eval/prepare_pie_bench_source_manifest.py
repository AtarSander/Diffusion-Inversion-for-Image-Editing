"""Create and freeze a manually labeled PIE-Bench source-group manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

_ALLOWED_LABELS = {"artificial_graphic", "photographic", "uncertain"}


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _write_jsonl(records: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_template(mapping: dict[str, dict]) -> list[dict]:
    image_paths = sorted({str(item["image_path"]) for item in mapping.values()})
    return [{"image_path": image_path, "label": "pending"} for image_path in image_paths]


def freeze_manifest(records: list[dict], expected_image_paths: set[str]) -> list[dict]:
    labels: dict[str, str] = {}
    for record in records:
        image_path = record.get("image_path")
        label = record.get("label")
        if not isinstance(image_path, str) or not isinstance(label, str):
            raise ValueError("Every manifest record requires string image_path and label fields.")
        if image_path in labels:
            raise ValueError(f"Duplicate image_path in source manifest: {image_path}")
        if label not in _ALLOWED_LABELS:
            raise ValueError(
                f"Unsupported label {label!r} for {image_path}; expected one of {sorted(_ALLOWED_LABELS)}."
            )
        labels[image_path] = label
    if set(labels) != expected_image_paths:
        missing = sorted(expected_image_paths - set(labels))
        extra = sorted(set(labels) - expected_image_paths)
        raise ValueError(
            f"Source manifest does not match PIE-Bench mapping; missing={missing[:3]}, extra={extra[:3]}."
        )
    return [{"image_path": path, "label": labels[path]} for path in sorted(labels)]


@hydra.main(
    config_path="../../config",
    config_name="eval/prepare_pie_bench_source_manifest",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    data_path = Path(to_absolute_path(str(cfg.data_path)))
    mapping_path = data_path / "mapping_file.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"PIE-Bench mapping file not found: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_image_paths = {str(item["image_path"]) for item in mapping.values()}
    mode = str(cfg.mode)
    if mode == "template":
        records = create_template(mapping)
        output_path = Path(to_absolute_path(str(cfg.template_path)))
        _write_jsonl(records, output_path)
        print(f"Wrote {len(records)} pending labels to {output_path}")
        return
    if mode != "freeze":
        raise ValueError("mode must be template or freeze")
    input_path = Path(to_absolute_path(str(cfg.labeled_path)))
    frozen_path = Path(to_absolute_path(str(cfg.frozen_path)))
    records = freeze_manifest(_read_jsonl(input_path), expected_image_paths)
    checksum = _write_jsonl(records, frozen_path)
    summary = {
        "num_images": len(records),
        "counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "sha256": checksum,
    }
    summary_path = frozen_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen source manifest: {frozen_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
