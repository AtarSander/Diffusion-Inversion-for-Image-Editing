"""Generate paper tables and plots for the matched SDXL inversion experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LORA_ROWS = [
    {"method": "LoRA $r=8$", "corr": 0.0999, "kl": 0.0767, "mae": 0.02063, "lpips": 0.05982, "psnr": 30.06, "ssim": 0.8654},
    {"method": "LoRA $r=16$", "corr": 0.0989, "kl": 0.0671, "mae": 0.02028, "lpips": 0.05774, "psnr": 30.22, "ssim": 0.8699},
    {"method": "LoRA $r=32$", "corr": 0.0972, "kl": 0.0504, "mae": 0.01970, "lpips": 0.05538, "psnr": 30.40, "ssim": 0.8743},
]

REPORTED_TADA_ROWS = [
    {"method": "DDIM Inv.", "corr": 0.16, "kl": 0.89, "mae": 0.03, "lpips": 0.10, "psnr": 27.58, "ssim": 0.81},
    {"method": "Forward $2\\%$", "corr": 0.14, "kl": 0.71, "mae": 0.03, "lpips": 0.10, "psnr": 27.12, "ssim": 0.79},
    {"method": "Forward $4\\%$", "corr": 0.09, "kl": 0.38, "mae": 0.04, "lpips": 0.14, "psnr": 25.68, "ssim": 0.72},
]

VARIANTS = [
    ("ddim", "DDIM Inv."),
    ("forward_2pct", "Forward $2\\%$"),
    ("forward_4pct", "Forward $4\\%$"),
]


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        current = current[key]
    return current


def load_matched_rows(results_root: Path) -> list[dict[str, float | str]] | None:
    rows: list[dict[str, float | str]] = []
    for variant, label in VARIANTS:
        path = results_root / variant / "evaluation_summary.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        aggregate = data["aggregate"]
        image = aggregate["final_vs_reconstructed_image"]
        rows.append(
            {
                "method": label,
                "corr": float(_nested(aggregate, "latent_normality", "corr")),
                "kl": 100.0 * float(_nested(aggregate, "latent_normality", "kl")),
                "mae": float(image["mae/mean"]),
                "lpips": float(image["lpips/mean"]),
                "psnr": float(image["psnr_db/mean"]),
                "ssim": float(image["ssim/mean"]),
            }
        )
    return rows


def _best(rows: list[dict[str, float | str]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return max(values) if metric in {"psnr", "ssim"} else min(values)


def _cell(value: float, best_value: float) -> str:
    rendered = f"{value:.2f}"
    return f"\\textbf{{{rendered}}}" if abs(value - best_value) < 1e-12 else rendered


def write_rows_tex(path: Path, tada_rows: list[dict[str, float | str]] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tada_rows is None:
        lines = [
            "Matched & DDIM Inv. & -- & -- & -- & -- & -- & -- \\\\",
            "Matched & Forward $2\\%$ & -- & -- & -- & -- & -- & -- \\\\",
            "Matched & Forward $4\\%$ & -- & -- & -- & -- & -- & -- \\\\",
            "\\midrule",
        ]
        for row in LORA_ROWS:
            lines.append(
                "Ours & {method} & {corr:.2f} & {kl:.2f} & {mae:.2f} & {lpips:.2f} & {psnr:.2f} & {ssim:.2f} \\\\".format(**row)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    all_rows = tada_rows + LORA_ROWS
    best = {metric: _best(all_rows, metric) for metric in ("corr", "kl", "mae", "lpips", "psnr", "ssim")}
    lines = []
    for idx, row in enumerate(all_rows):
        if idx == len(tada_rows):
            lines.append("\\midrule")
        source = "Matched" if idx < len(tada_rows) else "Ours"
        cells = [_cell(float(row[metric]), best[metric]) for metric in ("corr", "kl", "mae", "lpips", "psnr", "ssim")]
        lines.append(f"{source} & {row['method']} & " + " & ".join(cells) + " \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("reports/eval/sdxl_tada_matched"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/sdxl_tada_matched_rows.tex"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/generated/sdxl_tada_matched_metrics.json"))
    args = parser.parse_args()
    rows = load_matched_rows(args.results_root)
    write_rows_tex(args.output, rows)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"matched": rows, "lora": LORA_ROWS}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
