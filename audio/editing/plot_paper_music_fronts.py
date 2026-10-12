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

# "paper" is the published figure: AudioLDM2 at 400 NFE with guided inversion, Stable Audio
# before the sigma = 0 lift. "lift" is AudioLDM2 at 200 NFE with unguided inversion and Stable
# Audio with the lift, comparing the trajectory and real adapters (Stable Audio: both trained on
# the last sampling step too).
FIGURES = {
    "paper": {
        "AudioLDM2": {
            "csv": AUDIO_ROOT / "output/matched_nfe_audioldm2/20260925_213426_full_nfe400/matched_nfe_runs.csv",
            "labels": {
                "DDIM Inv.": "DDIM Inv.",
                "DDIM Inv. + LoRA (Gen)": "Moonwalker $\\mathbf{(ours)}$",
                "DDPM Inv.": "DDPM Inv.",
                "SDEdit": "SDEdit",
            },
        },
        "Stable Audio Open": {
            "csv": AUDIO_ROOT / "output/matched_nfe/20260926_023532/matched_nfe_runs_full.csv",
            "labels": {
                "ODEInv (no LoRA)": "DDIM Inv.",
                "ODEInv w/ LoRA bw": "Moonwalker $\\mathbf{(ours)}$",
                "DDPM-inv": "DDPM Inv.",
                "SDEdit": "SDEdit",
            },
        },
    },
    "lift": {
        "AudioLDM2": {
            "csv": AUDIO_ROOT / "output/matched_nfe_audioldm2/20261012_161241_full_nfe200/matched_nfe_runs.csv",
            # The cycle arm is lambda = 1000; lambda = 1 matches the trajectory adapter to 3e-5 LPAPS
            # at every setting, so it would only redraw that front.
            "labels": {
                "DDIM Inv. [cfg_src 1]": "DDIM Inv.",
                "DDPM Inv.": "DDPM Inv.",
                "SDEdit": "SDEdit",
                "DDIM Inv. + LoRA (Gen) [cfg_src 1]": "Moonwalker (traj.)",
                "DDIM Inv. + LoRA (Real) [cfg_src 1]": "Moonwalker (real)",
                "DDIM Inv. + LoRA (Cycle lambda=1000) [cfg_src 1]": "Moonwalker (cycle)",
            },
            "zoom": (4.10, 4.75),
        },
        "Stable Audio Open": {
            "csv": AUDIO_ROOT / "output/matched_nfe/20261012_154351/matched_nfe_runs_full.csv",
            "labels": {
                "ODEInv (no LoRA)": "DDIM Inv.",
                "DDPM-inv": "DDPM Inv.",
                "SDEdit": "SDEdit",
                "ODEInv w/ LoRA saocos_finalpair_r8_a4_lr5e-5": "Moonwalker (traj.)",
                "ODEInv w/ LoRA saocos_realfndurfinal_r8_a4_lr5e-5": "Moonwalker (real)",
                "ODEInv w/ LoRA saocos_cycfinal_k1_g1.0_r8_a4_lr5e-5": "Moonwalker (cycle)",
            },
            "zoom": (3.50, 4.15),
        },
    },
}

STYLE = {
    "DDIM Inv.": ("#1f77b4", "o"),
    "Moonwalker $\\mathbf{(ours)}$": ("#d62728", "s"),
    "Moonwalker (traj.)": ("#d62728", "s"),
    "Moonwalker (real)": ("#9467bd", "D"),
    "Moonwalker (cycle)": ("#8c564b", "P"),
    "DDPM Inv.": ("#2ca02c", "^"),
    "SDEdit": ("#ff7f0e", "v"),
}

METRICS = {"clap": "CLAP-T", "muq": "MuQ-T"}
# Arms redrawn in the zoom panels: the adapters and the inversion they modify.
ZOOM_LABELS = ("DDIM Inv.", "Moonwalker (traj.)", "Moonwalker (real)", "Moonwalker (cycle)")


def draw_zoom(ax: plt.Axes, axz: plt.Axes, fronts: dict, metric: str,
              window: tuple[float, float]) -> None:
    """Redraw the ZOOM_LABELS fronts of one panel inside an LPAPS window and outline it on `ax`.

    Args:
        ax: Full panel, which gets a dashed rectangle around the zoomed region.
        axz: Zoom panel.
        fronts: Label -> front rows (lpaps, metric and their SEMs) drawn on `ax`.
        metric: Alignment column on the y axis.
        window: LPAPS range shown in the zoom.
    """
    x0, x1 = window
    ys = []
    for label, front in fronts.items():
        if label not in ZOOM_LABELS:
            continue
        color, marker = STYLE[label]
        axz.errorbar(front.lpaps, front[metric], xerr=front.lpaps_sem, yerr=front[f"{metric}_sem"],
                     color=color, marker=marker, markersize=7, markeredgecolor="black",
                     markeredgewidth=0.8, linewidth=1.6, elinewidth=0.9, capsize=3)
        inside = front[front.lpaps.between(x0, x1)]
        ys += [inside[metric] - inside[f"{metric}_sem"], inside[metric] + inside[f"{metric}_sem"]]
    y = pd.concat(ys)
    pad = 0.05 * (y.max() - y.min())
    y0, y1 = y.min() - pad, y.max() + pad
    axz.set_xlim(x0, x1)
    axz.set_ylim(y0, y1)
    axz.grid(True, linestyle="--", alpha=0.2)
    axz.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    axz.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    axz.tick_params(labelsize=FS - 2)
    ax.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                              edgecolor="0.3", linestyle="--", linewidth=1))


def main(figure: str = "paper", zoom: bool = False,
         out_root: str = str(AUDIO_ROOT / "output/paper_figures"),
         paper_figures: str = str(AUDIO_ROOT.parent / "paper_tex/figures")) -> None:
    """Plot per-arm Pareto fronts over the pooled (depth, cfg_tar) grid, with all
    operating points as faint markers behind each front.

    Args:
        figure: Key of FIGURES to draw. Only "paper" without zoom is copied into the paper.
        zoom: Add a panel right of each model's panels with the DDIM and adapter fronts inside
            that model's "zoom" LPAPS window.
        out_root: Directory for the timestamped copy of the figure.
        paper_figures: Paper figures directory receiving the stable-named copy.
    """
    per_model = 2 if zoom else 1
    fig, axes = plt.subplots(2, 2 * per_model, figsize=(5.5 * 2 * per_model, 5.625))
    for col, (model, spec) in enumerate(FIGURES[figure].items()):
        df = pd.read_csv(spec["csv"])
        for row, (metric, metric_name) in enumerate(METRICS.items()):
            ax = axes[row, col * per_model]
            front_x = []
            fronts = {}
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
                front_x += [front.lpaps - front.lpaps_sem, front.lpaps + front.lpaps_sem]
                fronts[label] = front
            # Each panel spans its own fronts, so dominated points past the last front point of
            # any arm fall outside it.
            x_lo, x_hi = pd.concat(front_x).min(), pd.concat(front_x).max()
            margin = 0.03 * (x_hi - x_lo)
            ax.set_xlim(x_lo - margin, x_hi + margin)
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
            else:
                ax.tick_params(labelbottom=True)
            if zoom:
                axz = axes[row, col * per_model + 1]
                draw_zoom(ax, axz, fronts, metric, spec["zoom"])
                if row == 1:
                    axz.set_xlabel("LPAPS to source $\\downarrow$", fontsize=FS)
        axes[0, col * per_model].set_title(model, fontsize=FS + 2)
        if zoom:
            axes[0, col * per_model + 1].set_title(f"{model} (zoom)", fontsize=FS + 2)
    # Union over the panels: an arm may exist for one model only.
    legend = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, handle)
    fig.legend(legend.values(), legend.keys(), ncol=len(legend), fontsize=FS,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"{ts}_music_editing_fronts_{figure}{'_zoom' if zoom else ''}.pdf"
    fig.savefig(stamped, bbox_inches="tight")
    fig.savefig(stamped.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if figure != "paper" or zoom:
        print(f"wrote {stamped}")
        return

    stable = Path(paper_figures) / "music_editing_fronts.pdf"
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(stamped, stable)
    print(f"wrote {stamped}\ncopied to {stable}")


if __name__ == "__main__":
    fire.Fire(main)
