# ABOUTME: Matched-NFE comparison table and figure for AudioLDM2: five arms at one denoiser-call
# ABOUTME: budget, inversion depth (tstart/steps) as the front axis, LPAPS vs CLAP/MuQ.

import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT / "editing"))
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

SUBDIRS = ("audioldm2_ddim", "audioldm2_ddpm", "audioldm2_sdedit")
TAIL = r"_(?P<split>[a-z]+)_nfe(?P<nfe>\d+)_t(?P<tstart>\d+)_s(?P<steps>\d+)_cfgtar(?P<cfg_tar>[\d.]+)$"
BASE = re.compile(r"^audioldm2_(?P<mode>ddim|ddpm|sdedit)_nolora" + TAIL)
LORA = re.compile(r"^audioldm2_ddimlora_(?P<adapter>.+?)_checkpoint_step_\d+" + TAIL)

# Which adapter directory maps to which arm; the trajectory twin is "Gen", forward-noise is "Real".
ADAPTER_ARM = {"attn_r8_a4_lr5e-5": "DDIM Inv. + LoRA (Gen)",
               "aldm2realfn_r8_a4_lr5e-5": "DDIM Inv. + LoRA (Real)"}
MODE_ARM = {"ddim": "DDIM Inv.", "ddpm": "DDPM Inv.", "sdedit": "SDEdit"}
COLORS = {"DDIM Inv.": "#1f77b4", "DDIM Inv. + LoRA (Gen)": "#d62728",
          "DDIM Inv. + LoRA (Real)": "#9467bd", "DDPM Inv.": "#2ca02c", "SDEdit": "#ff7f0e"}
METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0"}
PANELS = [("clap", "Alignment = CLAP"), ("muq", "Alignment = MuQ")]
BASELINE = "DDIM Inv."
FS = 16


def collect(runs_root: Path, split: str, nfe: int) -> pd.DataFrame:
    """Read every scored AudioLDM2 matched-NFE run of one split and budget into one row each.

    Args:
        runs_root: Directory holding the `audioldm2_*` run subdirectories.
        split: Benchmark split in the run names (`hparam` or `full`).
        nfe: Denoiser-call budget in the run names.

    Returns:
        One row per run: arm label, depth, guidance, and the mean/SEM of each metric.
    """
    rows = []
    for subdir in SUBDIRS:
        for run_dir in sorted((runs_root / subdir).glob(f"audioldm2_*_{split}_nfe{nfe}_*")):
            m = LORA.match(run_dir.name) or BASE.match(run_dir.name)
            csv = run_dir / PER_EXAMPLE_CSV
            if m is None or not csv.exists():
                continue
            g = m.groupdict()
            arm = ADAPTER_ARM[g["adapter"]] if "adapter" in g else MODE_ARM[g["mode"]]
            tstart, steps = int(g["tstart"]), int(g["steps"])
            frame = pd.read_csv(csv)
            row = {"arm": arm, "tstart": tstart, "steps": steps, "nfe": int(g["nfe"]),
                   "depth": round(100 * tstart / steps), "cfg_tar": float(g["cfg_tar"]),
                   "n": len(frame), "run": run_dir.name}
            for name, col in METRICS.items():
                row[name] = frame[col].mean()
                row[f"{name}_sem"] = frame[col].sem()
            rows.append(row)
    assert rows, f"no scored {split} nfe{nfe} runs under {runs_root}"
    return pd.DataFrame(rows).sort_values(["arm", "cfg_tar", "depth"]).reset_index(drop=True)


def pareto_front(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Operating points no other point of the arm beats on both lower LPAPS and higher alignment."""
    sub = sub.sort_values(["lpaps", metric], ascending=[True, False])
    return sub[sub[metric] > sub[metric].cummax().shift(fill_value=-np.inf)]


def paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Each arm minus DDIM Inv. at the same (depth, cfg_tar); LPAPS lower is better."""
    base = df[df.arm == BASELINE].set_index(["depth", "cfg_tar"])[list(METRICS)]
    out = []
    for arm, sub in df[df.arm != BASELINE].groupby("arm"):
        d = sub.set_index(["depth", "cfg_tar"])[list(METRICS)] - base
        d = d.dropna()
        out.append({"arm": arm, "cells": len(d),
                    **{f"d_{m}_mean": d[m].mean() for m in METRICS},
                    "lpaps_better_cells": int((d["lpaps"] < 0).sum())})
    return pd.DataFrame(out)


def main(runs_root: str, split: str = "hparam", nfe: int = 800,
         out_root: str = "output/matched_nfe_audioldm2") -> None:
    """Write the AudioLDM2 matched-NFE table and the LPAPS-vs-alignment front figure.

    Args:
        runs_root: Directory holding `audioldm2_ddim/`, e.g. .../edits/medleymd.
        split: Benchmark split to plot (`hparam` or `full`).
        nfe: Denoiser-call budget to plot.
        out_root: Destination, relative to `audio/`.
    """
    df = collect(Path(runs_root), split, nfe)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = AUDIO_ROOT / out_root / f"{stamp}_{split}_nfe{nfe}"
    out.mkdir(parents=True, exist_ok=True)

    cfgs = sorted(df["cfg_tar"].unique())
    show = df[["arm", "cfg_tar", "depth", "tstart", "steps", "lpaps", "clap", "muq"]]
    print(df.groupby("arm").size().to_string())
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    delta = paired_delta(df)
    print("\nArm - DDIM Inv., paired at matched (depth, cfg_tar):")
    print(delta.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.6 * len(PANELS), 5.8),
                             layout="constrained")
    fig.suptitle(f"Audio editing with AudioLDM2 at matched NFEs (~{nfe} denoiser calls)",
                 fontsize=19, fontweight="bold")
    for ax, (metric, name) in zip(axes, PANELS):
        for arm, sub in df.groupby("arm"):
            color = COLORS.get(arm)
            ax.scatter(sub["lpaps"], sub[metric], s=22, color=color, alpha=0.25,
                       edgecolor="black", linewidth=0.4)
            front = pareto_front(sub, metric)
            ax.errorbar(front["lpaps"], front[metric], xerr=front["lpaps_sem"],
                        yerr=front[f"{metric}_sem"], marker="o", ms=7, lw=1.8, capsize=2,
                        color=color, markeredgecolor="black", markeredgewidth=0.8,
                        label=arm)
        ax.set_xlabel("LPAPS to source (lower = better preserved)", fontsize=FS)
        ax.set_ylabel(name + " (higher = better edit)", fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        ax.text(0.035, 0.955, "★", transform=ax.transAxes, fontsize=26, color="#f1c40f",
                ha="center", va="center")
        ax.grid(True, linestyle="--", alpha=0.2)
    axes[0].legend(fontsize=FS - 3, loc="lower right")
    for ext in ("png", "svg"):
        fig.savefig(out / f"matched_nfe_front.{ext}", dpi=150, bbox_inches="tight")

    lines = [f"# AudioLDM2 at matched NFE (~{nfe} denoiser calls), split={split}\n",
             f"{len(df)} runs, {df['n'].iloc[0]} edits each, cfg_tar pooled "
             f"({', '.join(f'{c:g}' for c in cfgs)}). Lines are each arm's Pareto front over "
             "all (depth, cfg_tar) cells; faint dots are every cell.\n",
             "Figure: `matched_nfe_front.png`.\n",
             f"\n## Arm - {BASELINE}, paired at matched (depth, cfg_tar)\n",
             delta.to_markdown(index=False, floatfmt="+.4f"),
             "\n\n## All runs\n", show.to_markdown(index=False, floatfmt=".4f")]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    df.to_csv(out / "matched_nfe_runs.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
