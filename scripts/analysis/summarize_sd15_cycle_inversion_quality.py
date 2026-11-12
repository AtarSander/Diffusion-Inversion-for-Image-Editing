"""Summarize paired SD1.5 standard- and cycle-loss inversion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    ("ddim", "DDIM Inv."),
    ("standard_r8", "Standard LoRA $r=8$"),
    ("standard_r16", "Standard LoRA $r=16$"),
    ("standard_r32", "Standard LoRA $r=32$"),
    ("cycle_r8", "Cycle LoRA $r=8$"),
    ("cycle_r16", "Cycle LoRA $r=16$"),
    ("cycle_r32", "Cycle LoRA $r=32$"),
)
METRICS = ("corr", "kl", "mae", "lpips", "psnr", "ssim")


def _load_row(path: Path, label: str) -> dict[str, float | str | int]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    aggregate = data["aggregate"]
    image = aggregate["final_vs_reconstructed_image"]
    return {
        "method": label,
        "num_samples": int(data["num_samples"]),
        "corr": float(aggregate["latent_normality"]["corr"]),
        "kl": 100.0 * float(aggregate["latent_normality"]["kl"]),
        "mae": float(image["mae/mean"]),
        "lpips": float(image["lpips/mean"]),
        "psnr": float(image["psnr_db/mean"]),
        "ssim": float(image["ssim/mean"]),
    }


def _format_row(label: str, row: dict[str, float | str | int] | None) -> str:
    if row is None:
        return f"{label} & -- & -- & -- & -- & -- & -- \\\\"
    cells = [f"{float(row[metric]):.4f}" for metric in METRICS]
    cells[4] = f"{float(row['psnr']):.2f}"
    return f"{label} & " + " & ".join(cells) + " \\\\"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("reports/eval/sd15_cycle_inversion_quality"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/generated/sd15_cycle_inversion_quality.json"),
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=Path("reports/generated/sd15_cycle_inversion_quality_rows.tex"),
    )
    args = parser.parse_args()

    rows: dict[str, dict[str, float | str | int] | None] = {}
    tex_rows: list[str] = []
    for variant, label in VARIANTS:
        result_path = args.results_root / variant / "evaluation_summary.json"
        row = _load_row(result_path, label) if result_path.exists() else None
        rows[variant] = row
        if variant == "cycle_r8":
            tex_rows.append("\\midrule")
        tex_rows.append(_format_row(label, row))

    payload = {
        "protocol": "sd15_in_sample_on_sdxl_heldout_prompt_ids",
        "warning": (
            "These prompts were included in SD1.5 LoRA training. Compare rows only "
            "within this paired SD1.5 diagnostic; do not mix them with the held-out "
            "SDXL inversion-quality table."
        ),
        "rows": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.tex_output.write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.tex_output}")


if __name__ == "__main__":
    main()
