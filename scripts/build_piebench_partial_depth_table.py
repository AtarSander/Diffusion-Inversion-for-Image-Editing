#!/usr/bin/env python3
"""Build the PIE-Bench inversion-depth comparison from metric summaries."""

import argparse
import json
import math
from pathlib import Path


EDITORS = (
    ("P2P", "p2p", "p2p_lora_eval_outputs"),
    ("PnP", "pnp", "pnp_lora_eval_outputs"),
    ("MasaCtrl", "masactrl", "masactrl_lora_eval_outputs"),
    ("Pix2Pix-Zero", "pix2pix-zero", "pix2pix_zero_lora_eval_outputs"),
)

METRICS = (
    ("structure_distance", 1000, 700, False),
    ("psnr_unedit_part", 1, 556, True),
    ("lpips_unedit_part", 1000, 556, False),
    ("mse_unedit_part", 10000, 556, False),
    ("ssim_unedit_part", 100, 556, True),
    ("clip_similarity_target_image", 1, 700, True),
    ("clip_similarity_target_image_edit_part", 1, 700, True),
)

# Rounded full-depth r16 LoRA + DirectInv means from reports/pnp_lora_results.tex.
# The underlying JSON summaries are no longer present with the image outputs.
FULL_DIRECT_LORA_MEANS = {
    "p2p": [14.30, 25.95, 73.20, 44.76, 83.48, 25.31, 22.39],
    "pnp": [24.28, 22.40, 110.24, 81.56, 79.88, 25.80, 22.86],
    "masactrl": [28.50, 21.89, 98.89, 99.89, 80.47, 24.62, 21.49],
    "pix2pix-zero": [51.62, 21.23, 147.11, 140.85, 77.04, 23.49, 21.02],
}


def read_means(path: Path, method: str) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("num_images") != 700:
        raise ValueError(f"Expected 700 images in {path}")

    values = []
    for metric, scale, count, _ in METRICS:
        item = payload["metrics"][f"{method}|{metric}"]
        if item["count"] != count or not math.isfinite(item["mean"]):
            raise ValueError(f"Invalid {metric} in {path}: {item}")
        values.append(item["mean"] * scale)
    return values


def metric_row(
    label: str,
    depth: str,
    values: list[float],
    best: list[float],
    direct_inv: list[float],
    highlight_complete_rows: bool,
) -> str:
    if len(values) != len(METRICS):
        raise ValueError(f"Expected {len(METRICS)} metrics, got {len(values)}")
    cells = []
    better_flags = []
    for value, best_value, reference, (_, _, _, higher_better) in zip(
        values, best, direct_inv, METRICS
    ):
        rendered = f"{value:.2f}"
        displayed_value = float(rendered)
        if displayed_value == best_value:
            rendered = rf"\textbf{{{rendered}}}"
        better_than_direct_inv = (
            displayed_value > reference if higher_better else displayed_value < reference
        )
        better_flags.append(better_than_direct_inv)
        if better_than_direct_inv:
            rendered = rf"\underline{{{rendered}}}"
        cells.append(rendered)
    cells_text = " & ".join(cells)
    prefix = r"\rowcolor{green!12}" if highlight_complete_rows and all(better_flags) else ""
    return f"{prefix}{label} & {depth} & {cells_text} " + r"\\"


def editor_table(
    name: str,
    key: str,
    full_dir: str,
    partial_root: Path,
    full_root: Path,
    baseline_root: Path,
    p2p_fp32_baseline_root: Path,
    partial_depths: list[int],
    label_prefix: str,
    highlight_complete_rows: bool,
) -> list[str]:
    baseline = p2p_fp32_baseline_root if key == "p2p" else baseline_root
    pure_method = f"lora+{key}"
    direct_method = f"lora+directinversion+{key}"
    baseline_rows = [
        (label, r"100\%", read_means(baseline / method / "metrics_summary.json", method))
        for method, label in ((f"ddim+{key}", "DDIM"), (f"directinversion+{key}", "DirectInv"))
    ]
    moonwalker_rows = [
        (
            "Moonwalker",
            rf"{percent}\%",
            read_means(partial_root / f"p{percent}" / "metrics" / f"{pure_method}_summary.json", pure_method),
        )
        for percent in partial_depths
    ]
    moonwalker_rows.append(
        ("Moonwalker", r"100\%", read_means(full_root / full_dir / "r16" / "metrics_summary.json", pure_method))
    )
    direct_rows = [
        (
            "Moonwalker + DirectInv",
            rf"{percent}\%",
            read_means(partial_root / f"p{percent}" / "metrics" / f"{direct_method}_summary.json", direct_method),
        )
        for percent in partial_depths
    ]
    direct_rows.append(("Moonwalker + DirectInv", r"100\%", FULL_DIRECT_LORA_MEANS[key]))
    groups = (baseline_rows, moonwalker_rows, direct_rows)
    all_rows = [row for group in groups for row in group]
    best = [
        (max if metric[3] else min)(float(f"{row[2][index]:.2f}") for row in all_rows)
        for index, metric in enumerate(METRICS)
    ]
    direct_inv = [float(f"{value:.2f}") for value in baseline_rows[1][2]]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{name}: inversion depth versus reproduced full-depth baselines. All values are means.}}",
        rf"\label{{tab:{label_prefix}-{key}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Inversion & Depth & Structure ($\times10^3$) $\downarrow$ & PSNR (dB) $\uparrow$ & LPIPS ($\times10^3$) $\downarrow$ & MSE ($\times10^4$) $\downarrow$ & SSIM ($\times10^2$) $\uparrow$ & CLIP target $\uparrow$ & CLIP edited $\uparrow$ \\",
    ]

    for group in groups:
        lines.append(r"\midrule")
        for label, depth, values in group:
            lines.append(metric_row(label, depth, values, best, direct_inv, highlight_complete_rows))
    lines.extend((r"\bottomrule", r"\end{tabular}}", r"\end{table}"))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial-root", type=Path,
        default=Path("data/evaluation/pnp_lora_partial_inversion_r16"),
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--p2p-fp32-baseline-root", type=Path, required=True)
    parser.add_argument(
        "--depths", nargs="+", type=int, default=list(range(10, 100, 10)),
        help="Partial inversion depths to include, as distinct even percentages below 100",
    )
    parser.add_argument("--label-prefix", default="partial-depth")
    parser.add_argument("--title", default="PIE-Bench Moonwalker Inversion-Depth Comparison")
    parser.add_argument("--highlight-complete-rows", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/piebench_partial_inversion_depth_comparison.tex"),
    )
    args = parser.parse_args()
    if len(set(args.depths)) != len(args.depths) or any(
        percent < 2 or percent >= 100 or percent % 2 for percent in args.depths
    ):
        parser.error("--depths must contain distinct even percentages from 2 to 98")

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=16mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage[table]{xcolor}",
        rf"\title{{{args.title}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent All rows use the 700-image PIE-Bench set. Unedited-region metrics "
        r"use 556 images; the other metrics use 700. The partial-depth runs use the "
        r"same rank-16 SD1.5 Moonwalker checkpoint and 50-step DDIM schedule, stopping "
        r"inversion after the stated fraction and denoising from that point. "
        r"DDIM and DirectInv are reproduced baselines; P2P uses the FP32 "
        r"reproduction. Values are not original-paper numbers. Within each editor, "
        r"bold marks the best displayed value for each metric; underlining marks "
        r"values strictly better than the reproduced DirectInv baseline for that metric. "
        r"The 100\% Moonwalker + DirectInv means are rounded values from the earlier "
        r"PIE-Bench report because their original JSON summaries are unavailable.",
        *(
            [r"\par\noindent Light green marks rows where all seven metrics improve "
             r"strictly over reproduced DirectInv at the displayed precision."]
            if args.highlight_complete_rows else []
        ),
        r"\par\medskip",
    ]

    for index, spec in enumerate(EDITORS):
        if index:
            lines.append(r"\clearpage")
        lines.extend(
            editor_table(
                *spec,
                args.partial_root,
                args.full_root,
                args.baseline_root,
                args.p2p_fp32_baseline_root,
                args.depths,
                args.label_prefix,
                args.highlight_complete_rows,
            )
        )

    lines.append(r"\end{document}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
