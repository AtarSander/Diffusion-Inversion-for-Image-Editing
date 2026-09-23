# ABOUTME: Renders the paper figure for the Music Editing section: LPAPS-CLAP and
# ABOUTME: LPAPS-MuQ fronts for AudioLDM2 and Stable Audio Open from matched-NFE CSVs.

import shutil
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
FS = 14

MODELS = {
    "AudioLDM2": {
        "csv": AUDIO_ROOT / "output/matched_nfe_audioldm2/20260923_102222/matched_nfe_runs.csv",
        "labels": {
            "DDIM Inv.": "DDIM Inv.",
            "DDIM Inv. + LoRA (Gen)": "DDIM Inv. + LoRA",
            "DDPM Inv.": "DDPM Inv.",
            "SDEdit": "SDEdit",
        },
    },
    "Stable Audio Open": {
        "csv": AUDIO_ROOT / "output/matched_nfe/20260915_120122/matched_nfe_runs_hparam.csv",
        "labels": {
            "ODEInv (no LoRA)": "ODE Inv.",
            "ODEInv w/ LoRA bw": "ODE Inv. + LoRA",
            "DDPM-inv": "DDPM Inv.",
            "SDEdit": "SDEdit",
        },
    },
}

STYLE = {
    "DDIM Inv.": ("#1f77b4", "o"),
    "ODE Inv.": ("#1f77b4", "o"),
    "DDIM Inv. + LoRA": ("#d62728", "s"),
    "ODE Inv. + LoRA": ("#d62728", "s"),
    "DDPM Inv.": ("#2ca02c", "^"),
    "SDEdit": ("#ff7f0e", "v"),
}

METRICS = {"clap": "CLAP-T", "muq": "MuQ-T"}


def main(out_root: str = str(AUDIO_ROOT / "output/paper_figures"),
         paper_figures: str = str(AUDIO_ROOT.parent / "paper_tex/figures")) -> None:
    """Plot per-arm Pareto fronts over the pooled (depth, cfg_tar) grid, with all
    operating points as faint markers behind each front.

    Args:
        out_root: Directory for the timestamped copy of the figure.
        paper_figures: Paper figures directory receiving the stable-named copy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    for col, (model, spec) in enumerate(MODELS.items()):
        df = pd.read_csv(spec["csv"])
        for row, (metric, metric_name) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for arm, label in spec["labels"].items():
                color, marker = STYLE[label]
                sub = df[df.arm == arm].sort_values("lpaps")
                front = sub[sub[metric].cummax().eq(sub[metric])]
                ax.scatter(sub.lpaps, sub[metric], color=color, marker=marker,
                           s=22, alpha=0.35, linewidths=0)
                ax.errorbar(front.lpaps, front[metric],
                            xerr=front.lpaps_sem, yerr=front[f"{metric}_sem"],
                            color=color, marker=marker, markersize=6,
                            markeredgecolor="black", markeredgewidth=0.8,
                            linewidth=1.6, elinewidth=0.9, capsize=2, label=label)
            ax.grid(True, linestyle="--", alpha=0.2)
            lo, hi = ax.get_ylim()
            span = hi - lo
            ax.set_yticks(np.round(np.linspace(lo + 0.05 * span, hi - 0.05 * span, 4), 2))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(labelsize=FS)
            if col == 0:
                ax.set_ylabel(f"{metric_name} $\\uparrow$", fontsize=FS)
            if row == 1:
                ax.set_xlabel("LPAPS to source $\\downarrow$", fontsize=FS)
        axes[0, col].set_title(model, fontsize=FS + 2)
        axes[0, col].legend(fontsize=FS - 1, framealpha=0.9)
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"{ts}_music_editing_fronts.pdf"
    fig.savefig(stamped, bbox_inches="tight")
    fig.savefig(stamped.with_suffix(".png"), dpi=200, bbox_inches="tight")

    stable = Path(paper_figures) / "music_editing_fronts.pdf"
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(stamped, stable)
    print(f"wrote {stamped}\ncopied to {stable}")


if __name__ == "__main__":
    fire.Fire(main)
