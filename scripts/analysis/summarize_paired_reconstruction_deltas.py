#!/usr/bin/env python3
"""Summarize paired per-image reconstruction changes versus Direct Inversion."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

METRICS = {
    "structure_dino_scaled_correctly": False,
    "psnr_valid": True,
    "lpips_valid_tiles": False,
    "mse_valid": False,
    "ssim_eroded": True,
}


def finite(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], probability: float) -> float:
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * probability)))]


def bootstrap_mean_interval(values: list[float], draws: int, seed: str) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = []
    for editor_dir in sorted(path for path in args.metrics_root.iterdir() if path.is_dir()):
        with (editor_dir / "per_image.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_method: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], {})[row["file_id"]] = row
        direct = by_method.pop("directinversion")
        for method, candidate in sorted(by_method.items()):
            shared_ids = sorted(set(direct) & set(candidate))
            for metric, higher_is_better in METRICS.items():
                raw_changes = []
                for file_id in shared_ids:
                    baseline = finite(direct[file_id].get(metric))
                    value = finite(candidate[file_id].get(metric))
                    if baseline is None or value is None:
                        continue
                    raw_changes.append(value - baseline)
                direction = 1.0 if higher_is_better else -1.0
                improvements = [direction * value for value in raw_changes]
                low, high = bootstrap_mean_interval(
                    improvements, args.bootstrap_draws,
                    f"{args.seed}:{editor_dir.name}:{method}:{metric}",
                )
                results.append({
                    "editor": editor_dir.name,
                    "candidate": method,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "n": len(improvements),
                    "mean_signed_improvement": sum(improvements) / len(improvements),
                    "std_signed_improvement": (
                        sum((value - sum(improvements) / len(improvements)) ** 2
                            for value in improvements) / len(improvements)
                    ) ** 0.5,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "improved_count": sum(value > 0 for value in improvements),
                    "tied_count": sum(value == 0 for value in improvements),
                    "improved_fraction": sum(value > 0 for value in improvements) / len(improvements),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    with (args.output / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
