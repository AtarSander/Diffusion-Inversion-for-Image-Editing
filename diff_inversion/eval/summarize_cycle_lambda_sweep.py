#!/usr/bin/env python3
"""Create one TeX table for the SD1.5 r16 cycle-loss/DirectInv sweep."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metric:
    key: str
    heading: str
    scale: float
    higher_is_better: bool


METRICS = (
    Metric("structure_distance", r"Structure ($\times 10^3$) $\downarrow$", 1_000.0, False),
    Metric("psnr_unedit_part", r"PSNR $\uparrow$", 1.0, True),
    Metric("lpips_unedit_part", r"LPIPS ($\times 10^3$) $\downarrow$", 1_000.0, False),
    Metric("mse_unedit_part", r"MSE ($\times 10^4$) $\downarrow$", 10_000.0, False),
    Metric("ssim_unedit_part", r"SSIM ($\times 10^2$) $\uparrow$", 100.0, True),
    Metric("clip_similarity_target_image", r"CLIP target $\uparrow$", 1.0, True),
    Metric("clip_similarity_target_image_edit_part", r"CLIP edit $\uparrow$", 1.0, True),
)
EDITORS = (
    ("P2P", "p2p"),
    ("PnP", "pnp"),
    ("MasaCtrl", "masactrl"),
    ("Pix2Pix-Zero", "pix2pix-zero"),
)
LAMBDA_ROOTS = (
    ("0.5", "cycle_loss_cfg1_r16_lambda0p5_uniform_e1_all_editors"),
    ("1", "cycle_loss_cfg1_r16_uniform_e1_all_editors"),
    ("2", "cycle_loss_cfg1_r16_lambda2_uniform_e1_all_editors"),
    ("4", "cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors"),
    ("8", "cycle_loss_cfg1_r16_lambda8_uniform_e1_all_editors"),
    ("16", "cycle_loss_cfg1_r16_lambda16_uniform_e1_all_editors"),
)
_KEYS = tuple(metric.key for metric in METRICS)
PAPER_DIRECTINV = {
    "p2p": dict(zip(_KEYS, (11.65, 27.22, 54.55, 32.86, 84.76, 25.02, 22.10), strict=True)),
    "pnp": dict(zip(_KEYS, (24.29, 22.46, 106.06, 80.45, 79.68, 25.41, 22.62), strict=True)),
    "masactrl": dict(zip(_KEYS, (24.70, 22.64, 87.94, 81.09, 81.33, 24.38, 21.35), strict=True)),
    "pix2pix-zero": dict(zip(_KEYS, (49.22, 21.53, 138.98, 127.32, 77.05, 23.31, 21.05), strict=True)),
}


def _load_row(evaluation_root: Path, root_name: str, method: str, *, allow_missing: bool):
    path = evaluation_root / root_name / "metrics" / f"{method}_summary.json"
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing metric summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("num_images", 0)) != 700:
        raise ValueError(f"Expected 700 images in {path}, got {payload.get('num_images')!r}")
    row = {}
    for metric in METRICS:
        entry = payload["metrics"][f"{method}|{metric.key}"]
        row[metric.key] = {
            "mean": float(entry["mean"]) * metric.scale,
            "std": float(entry["std"]) * metric.scale,
        }
    return row


def _best(rows, paper):
    result = {}
    for metric in METRICS:
        values = {"paper": paper[metric.key]}
        values.update({key: row[metric.key]["mean"] for key, row in rows.items() if row is not None})
        optimum = max(values.values()) if metric.higher_is_better else min(values.values())
        result[metric.key] = {key for key, value in values.items() if abs(value - optimum) <= 5e-10}
    return result


def _best_lambdas(rows):
    result = {}
    for metric in METRICS:
        values = {
            key: row[metric.key]["mean"]
            for key, row in rows.items()
            if row is not None
        }
        if not values:
            result[metric.key] = set()
            continue
        optimum = max(values.values()) if metric.higher_is_better else min(values.values())
        result[metric.key] = {
            key for key, value in values.items() if abs(value - optimum) <= 5e-10
        }
    return result


def _paper_cell(value: float, bold: bool) -> str:
    body = f"{value:.2f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return "$" + body + "$"


def _result_cell(
    entry, metric: Metric, paper_value: float, *, bold: bool, italic: bool
) -> str:
    body = rf"{entry['mean']:.2f} \pm {entry['std']:.2f}"
    if italic:
        body = rf"\mathit{{{body}}}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    beats_paper = entry["mean"] > paper_value if metric.higher_is_better else entry["mean"] < paper_value
    if beats_paper:
        body = rf"\underline{{{body}}}"
    return "$" + body + "$"


def build_tex(evaluation_root: Path, *, allow_missing: bool) -> str:
    table_rows = []
    for editor_index, (editor_name, editor_key) in enumerate(EDITORS):
        method = f"lora+directinversion+{editor_key}"
        rows = {
            value: _load_row(evaluation_root, root, method, allow_missing=allow_missing)
            for value, root in LAMBDA_ROOTS
        }
        paper = PAPER_DIRECTINV[editor_key]
        best = _best(rows, paper)
        best_lambdas = _best_lambdas(rows)
        if editor_index:
            table_rows.append(r"    \midrule")
        paper_cells = [_paper_cell(paper[m.key], "paper" in best[m.key]) for m in METRICS]
        table_rows.append("    " + " & ".join([editor_name, "Paper DirectInv", "--", *paper_cells]) + r" \tabularnewline")
        for value, _ in LAMBDA_ROOTS:
            row = rows[value]
            prefix = r"    \rowcolor{cyclebaseline} " if value == "1" else "    "
            cells = [""] * len(METRICS) if row is None else [
                _result_cell(
                    row[m.key],
                    m,
                    paper[m.key],
                    bold=value in best[m.key],
                    italic=value in best_lambdas[m.key],
                )
                for m in METRICS
            ]
            table_rows.append(prefix + " & ".join([editor_name, "Cycle LoRA+DirectInv", value, *cells]) + r" \tabularnewline")

    headings = " & ".join(["Editor", "Setup", r"$\lambda_{\mathrm{cycle}}$", *[m.heading for m in METRICS]])
    return "\n".join([
        r"\documentclass{article}",
        r"\usepackage[margin=1.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{cyclebaseline}{RGB}{232,242,255}",
        r"\begin{document}",
        r"\begin{table*}[htbp]",
        r"  \centering\scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{0.90}",
        r"  \caption{Cycle-loss LoRA+DirectInv r16 sweep across all editors. Paper DirectInv rows are published means without standard deviations; experimental rows are SD1.5 PIE-Bench mean $\pm$ standard deviation. Bold marks the best available value per editor and metric. Italics mark the best completed lambda value within each editor and metric. Underlining marks an experimental result better than Paper DirectInv. The shaded row is the previous $\lambda_{\mathrm{cycle}}=1$ run. Blank cells denote unfinished evaluations. Structure and LPIPS are scaled by $10^3$, MSE by $10^4$, and SSIM by $10^2$.}",
        r"  \label{tab:sd15-cycle-lambda-r16-all-editors}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lllrrrrrrr}",
        r"    \toprule",
        "    " + headings + r" \tabularnewline",
        r"    \midrule",
        *table_rows,
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
        r"\end{document}",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    evaluation_root = Path(os.path.realpath(repo_root / "checkpoints")).parent / "evaluation" / "one_epoch"
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_tex(evaluation_root, allow_missing=args.allow_missing), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
