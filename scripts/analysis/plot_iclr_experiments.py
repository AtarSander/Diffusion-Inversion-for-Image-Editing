"""Generate compact ICLR figures from the inversion and editing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.summarize_sdxl_tada_matched import (
    LORA_ROWS,
    REPORTED_TADA_ROWS,
    load_matched_rows,
)


STEP_VALUES = [0, 1, 2, 3, 4, 5, 50]
STEP_DATA = {
    8: {
        "corr": [0.1601, 0.1590, 0.1204, 0.1129, 0.1097, 0.1078, 0.0999],
        "lpips": [0.09617, 0.09546, 0.08507, 0.08416, 0.08391, 0.08393, 0.05982],
    },
    16: {
        "corr": [0.1601, 0.1588, 0.1180, 0.1105, 0.1076, 0.1059, 0.0989],
        "lpips": [0.09617, 0.09551, 0.08475, 0.08366, 0.08340, 0.08346, 0.05774],
    },
    32: {
        "corr": [0.1601, 0.1588, 0.1143, 0.1071, 0.1043, 0.1028, 0.0972],
        "lpips": [0.09617, 0.09536, 0.08402, 0.08283, 0.08250, 0.08249, 0.05538],
    },
}

EDITORS = ["P2P", "PnP", "MasaCtrl", "Pix2Pix-Zero"]
METRICS = ["Structure", "PSNR", "LPIPS", "MSE", "SSIM", "CLIP target", "CLIP edit"]
DIRECT = np.array([
    [31.35, 22.57, 125.51, 91.27, 78.54, 25.07, 22.22],
    [23.33, 22.46, 105.51, 79.97, 80.22, 25.63, 22.67],
    [23.57, 22.82, 86.83, 79.19, 82.02, 24.51, 21.47],
    [49.99, 21.47, 142.46, 133.49, 77.49, 23.53, 21.09],
])
FUSION = np.array([
    [14.01, 26.03, 72.35, 43.83, 83.56, 25.32, 22.31],
    [24.19, 22.38, 110.03, 81.64, 79.92, 25.82, 22.89],
    [28.60, 21.91, 98.73, 98.95, 80.54, 24.60, 21.50],
    [51.19, 21.36, 145.33, 138.62, 77.12, 23.40, 20.97],
])
HIGHER_BETTER = np.array([False, True, False, False, True, True, True])


def plot_pareto(output: Path, tada_rows, matched: bool) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.3))
    tada_offsets = [(7, 3), (7, 4), (7, -10)]
    for row, offset in zip(tada_rows, tada_offsets, strict=True):
        ax.scatter(
            row["corr"],
            row["lpips"],
            marker="^",
            s=60,
            facecolors="#D55E00" if matched else "none",
            edgecolors="#D55E00",
        )
        label = str(row["method"]).replace("$", "").replace("\\%", "%")
        ax.annotate(label, (row["corr"], row["lpips"]), xytext=offset, textcoords="offset points", fontsize=7)
    lora_offsets = [(8, 5), (8, -1), (8, -7)]
    for row, offset in zip(LORA_ROWS, lora_offsets, strict=True):
        ax.scatter(row["corr"], row["lpips"], marker="o", s=50, color="#0072B2")
        label = str(row["method"]).replace("$", "")
        ax.annotate(label, (row["corr"], row["lpips"]), xytext=offset, textcoords="offset points", fontsize=7)
    ax.set_xlabel("Recovered-noise patch correlation (lower is better)")
    ax.set_ylabel("Reconstruction LPIPS (lower is better)")
    ax.grid(alpha=0.25)
    ax.margins(x=0.05, y=0.16)
    ax.set_title("Normality--fidelity trade-off", pad=9)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_steps(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(STEP_VALUES))
    for rank, values in STEP_DATA.items():
        axes[0].plot(x, values["corr"], marker="o", label=f"r={rank}")
        axes[1].plot(x, values["lpips"], marker="o", label=f"r={rank}")
    for ax, ylabel in zip(axes, ["Patch correlation", "Reconstruction LPIPS"]):
        ax.set_xticks(x, [str(value) for value in STEP_VALUES])
        ax.set_xlabel("LoRA-active inversion steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_editing(output: Path) -> None:
    signed = (FUSION - DIRECT) / np.maximum(np.abs(DIRECT), 1e-12) * 100.0
    signed[:, ~HIGHER_BETTER] *= -1
    limit = max(20.0, float(np.nanmax(np.abs(signed))))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    image = ax.imshow(signed, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)), METRICS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(EDITORS)), EDITORS)
    for row in range(signed.shape[0]):
        for col in range(signed.shape[1]):
            value = signed[row, col]
            ax.text(col, row, f"{value:+.1f}%", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.45 * limit else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Improvement over DirectInv (%)")
    ax.set_title("Effect of adding LoRA inversion to Direct Inversion")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("reports/eval/sdxl_tada_matched"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_rows = load_matched_rows(args.results_root)
    plot_pareto(args.output_dir / "inversion_normality_fidelity.pdf", matched_rows or REPORTED_TADA_ROWS, matched_rows is not None)
    plot_steps(args.output_dir / "inversion_active_steps.pdf")
    plot_editing(args.output_dir / "editing_directinv_lora_relative_change.pdf")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
