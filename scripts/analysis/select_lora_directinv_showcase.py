#!/usr/bin/env python3
"""Rank paired PIE-Bench gains of LoRA+DirectInv over DirectInv.

The script never mixes checkpoints within an editor.  Each ``--candidate``
argument fixes one result root for one editor before images are ranked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "structure_distance": False,
    "psnr_unedit_part": True,
    "lpips_unedit_part": False,
    "mse_unedit_part": False,
    "ssim_unedit_part": True,
    "clip_similarity_target_image": True,
    "clip_similarity_target_image_edit_part": True,
}


def finite(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    prefix = next(column.split("|", 1)[0] for column in rows[0] if "|" in column)
    return {
        row["file_id"]: {
            metric: finite(row.get(f"{prefix}|{metric}")) for metric in METRICS
        }
        for row in rows
    }


def parse_candidate(value: str) -> tuple[str, str, Path]:
    try:
        editor, label, root = value.split("=", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate must have the form EDITOR=LABEL=RESULT_ROOT"
        ) from error
    return editor, label, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    mapping = json.loads((args.data_root / "mapping_file.json").read_text())
    editor_rows: dict[str, dict[str, dict]] = {}
    provenance = {}

    for editor, label, candidate_root in args.candidate:
        baseline_csv = args.baseline_root / "metrics" / f"directinversion+{editor}.csv"
        candidate_csv = candidate_root / "metrics" / f"lora+directinversion+{editor}.csv"
        baseline = read_metrics(baseline_csv)
        candidate = read_metrics(candidate_csv)
        shared = sorted(set(baseline) & set(candidate))
        rows: dict[str, dict] = {}
        for file_id in shared:
            item = mapping[file_id]
            changes = {}
            for metric, higher_is_better in METRICS.items():
                old = baseline[file_id][metric]
                new = candidate[file_id][metric]
                if old is None or new is None:
                    continue
                signed_gain = (new - old) if higher_is_better else (old - new)
                changes[metric] = {
                    "directinv": old,
                    "lora_directinv": new,
                    "signed_gain": signed_gain,
                }
            rows[file_id] = {
                "file_id": file_id,
                "image_path": item["image_path"],
                "editing_type_id": item["editing_type_id"],
                "original_prompt": item["original_prompt"],
                "editing_prompt": item["editing_prompt"],
                "editing_instruction": item["editing_instruction"],
                "metrics": changes,
            }

        # Convert incomparable metric units into empirical gain percentiles.
        for metric in METRICS:
            ranked = sorted(
                (
                    row["metrics"][metric]["signed_gain"],
                    file_id,
                )
                for file_id, row in rows.items()
                if metric in row["metrics"]
            )
            denominator = max(len(ranked) - 1, 1)
            for rank, (_, file_id) in enumerate(ranked):
                rows[file_id]["metrics"][metric]["gain_percentile"] = rank / denominator

        for row in rows.values():
            positive = [
                (values["gain_percentile"], metric)
                for metric, values in row["metrics"].items()
                if values["signed_gain"] > 0
            ]
            positive.sort(reverse=True)
            row["best_positive_metric"] = positive[0][1] if positive else None
            row["best_positive_percentile"] = positive[0][0] if positive else None

        editor_rows[editor] = rows
        provenance[editor] = {
            "label": label,
            "baseline_csv": str(baseline_csv),
            "candidate_csv": str(candidate_csv),
            "baseline_images": str(
                args.baseline_root / f"directinversion+{editor}" / "annotation_images"
            ),
            "candidate_images": str(
                candidate_root / f"lora+directinversion+{editor}" / "annotation_images"
            ),
        }

    per_editor = {}
    per_editor_consistent = {}
    for editor, rows in editor_rows.items():
        per_editor[editor] = {}
        for metric in METRICS:
            ranked = sorted(
                (row for row in rows.values() if metric in row["metrics"]),
                key=lambda row: row["metrics"][metric]["signed_gain"],
                reverse=True,
            )
            per_editor[editor][metric] = ranked[: args.top_k]
        consistent = []
        for row in rows.values():
            gains = [values for values in row["metrics"].values()]
            positive = [values for values in gains if values["signed_gain"] > 0]
            entry = dict(row)
            entry["positive_metric_count"] = len(positive)
            entry["available_metric_count"] = len(gains)
            entry["mean_positive_gain_percentile"] = (
                sum(values["gain_percentile"] for values in positive) / len(positive)
                if positive
                else None
            )
            consistent.append(entry)
        consistent.sort(
            key=lambda row: (
                row["positive_metric_count"],
                row["mean_positive_gain_percentile"] or -1,
            ),
            reverse=True,
        )
        per_editor_consistent[editor] = consistent[: args.top_k]

    shared_ids = set.intersection(*(set(rows) for rows in editor_rows.values()))
    cross_editor = []
    for file_id in shared_ids:
        best_by_editor = {
            editor: {
                "metric": rows[file_id]["best_positive_metric"],
                "gain_percentile": rows[file_id]["best_positive_percentile"],
            }
            for editor, rows in editor_rows.items()
            if rows[file_id]["best_positive_metric"] is not None
        }
        template = next(iter(editor_rows.values()))[file_id]
        cross_editor.append(
            {
                "file_id": file_id,
                "image_path": template["image_path"],
                "editing_type_id": template["editing_type_id"],
                "original_prompt": template["original_prompt"],
                "editing_prompt": template["editing_prompt"],
                "editing_instruction": template["editing_instruction"],
                "editors_improved": len(best_by_editor),
                "mean_best_gain_percentile": (
                    sum(entry["gain_percentile"] for entry in best_by_editor.values())
                    / len(best_by_editor)
                    if best_by_editor
                    else None
                ),
                "best_by_editor": best_by_editor,
                "metrics_by_editor": {
                    editor: rows[file_id]["metrics"]
                    for editor, rows in editor_rows.items()
                },
            }
        )
    cross_editor.sort(
        key=lambda row: (
            row["editors_improved"],
            row["mean_best_gain_percentile"] or -1,
        ),
        reverse=True,
    )

    output = {
        "metric_direction": {
            metric: "higher" if higher else "lower" for metric, higher in METRICS.items()
        },
        "provenance": provenance,
        "per_editor_top": per_editor,
        "per_editor_consistent_top": per_editor_consistent,
        "cross_editor_top": cross_editor[: args.top_k],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
