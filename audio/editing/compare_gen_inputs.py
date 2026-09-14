# ABOUTME: The H1 comparison figure: real-audio (hparam) vs generated-input (genhparam) fronts
# ABOUTME: for every method, and the paired LoRA-vs-no-LoRA delta per tstart on both splits.

import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))
sys.path.insert(0, str(AUDIO_ROOT / "editing"))
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

TSTARTS = [25, 50, 75, 99]
METHODS = {
    "DDPM-inv": "stableaudio_ddpm_{split}_cfgtar3.5_t{t}_s100",
    "SDEdit": "stableaudio_sdedit_{split}_cfgtar3.5_t{t}_s100",
    "ODEInv (no LoRA)": "stableaudio_odeinv_nolora_{split}_cfgtar3.5_t{t}_s100",
    "ODEInv + LoRA": (
        "stableaudio_odeinvlora_{split}_saocos_r8_a4_lr5e-5"
        "_checkpoint_step_4000_cfgtar3.5_t{t}_s100"
    ),
}
COLORS = {"DDPM-inv": "#2ca02c", "SDEdit": "#ff7f0e", "ODEInv (no LoRA)": "#7f7f7f",
          "ODEInv + LoRA": "#9467bd"}
DELTA_METRICS = {"lpaps": "LPAPS", "clap": "CLAP", "muqt_sim_p0": "MuQ", "mulan_dir": "Dir. MuLan"}
SPLIT_STYLE = {"hparam": ("o", "-", 1.0), "genhparam": ("s", "--", 0.8)}
SPLIT_LABEL = {"hparam": "real audio", "genhparam": "generated inputs"}
FS = 14


def frame(root: Path, split: str, method: str, t: int) -> pd.DataFrame:
    """Load one run's per-example table."""
    path = root / METHODS[method].format(split=split, t=t) / PER_EXAMPLE_CSV
    df = pd.read_csv(path)
    assert len(df) == 115, (path, len(df))
    return df


def main(runs_root: str, out_root: str = "output/gen_inputs") -> None:
    """Write the H1 figure (fronts + paired deltas) and its markdown mirror.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
    """
    root = Path(runs_root)

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(
        "H1 — editing model generations instead of real audio (cfg_tar 3.5, 100-step grid, "
        "115 paired rows)", fontsize=17, fontweight="bold", y=0.99,
    )

    # Top row: the fronts on both splits, one panel per alignment metric.
    for ax, (metric, name) in zip(
        axes[0], [("clap", "CLAP"), ("muqt_sim_p0", "MuQ-MuLan"),
                  ("clap_dir", "Directional CLAP"), ("mulan_dir", "Directional MuLan")]
    ):
        for split, (marker, style, alpha) in SPLIT_STYLE.items():
            for method, color in COLORS.items():
                xs, ys = [], []
                for t in TSTARTS:
                    df = frame(root, split, method, t)
                    xs.append(df["lpaps"].mean())
                    ys.append(df[metric].mean())
                ax.plot(xs, ys, style, marker=marker, ms=8, lw=1.8, color=color, alpha=alpha,
                        markeredgecolor="black", markeredgewidth=0.8,
                        label=f"{method} ({SPLIT_LABEL[split]})")
        ax.set_xlabel("LPAPS to source (lower = preserved)", fontsize=FS)
        ax.set_ylabel(f"{name} to target", fontsize=FS)
        ax.set_title(name, fontsize=FS + 1)
        ax.tick_params(labelsize=FS)
        ax.grid(True, linestyle="--", alpha=0.2)
    axes[0][0].legend(fontsize=FS - 3, loc="lower right", framealpha=0.9)

    # Bottom row: the H1 test itself -- the paired LoRA - no-LoRA delta per tstart, both splits.
    rows = []
    for split in SPLIT_STYLE:
        for t in TSTARTS:
            base = frame(root, split, "ODEInv (no LoRA)", t)
            lora = frame(root, split, "ODEInv + LoRA", t)
            assert (base["position"] == lora["position"]).all()
            for metric in DELTA_METRICS:
                d = lora[metric] - base[metric]
                ci = d.sem() * stats.t.ppf(0.975, len(d) - 1)
                rows.append({"split": split, "tstart": t, "metric": metric,
                             "delta": d.mean(), "ci95": ci,
                             "p": stats.ttest_rel(lora[metric], base[metric]).pvalue})
    deltas = pd.DataFrame(rows)

    width = 0.38
    for ax, (metric, name) in zip(axes[1], DELTA_METRICS.items()):
        for k, (split, _) in enumerate(SPLIT_STYLE.items()):
            sub = deltas[(deltas.split == split) & (deltas.metric == metric)]
            x = [i + (k - 0.5) * width for i in range(len(TSTARTS))]
            color = "#c44e52" if split == "genhparam" else "#4c72b0"
            ax.bar(x, sub["delta"], width, yerr=sub["ci95"], capsize=4,
                   color=color, edgecolor="black", linewidth=0.8,
                   label=SPLIT_LABEL[split])
            for xi, (_, r) in zip(x, sub.iterrows()):
                if r["p"] < 0.001:
                    star = "***"
                elif r["p"] < 0.01:
                    star = "**"
                elif r["p"] < 0.05:
                    star = "*"
                else:
                    star = ""
                if star:
                    off = r["ci95"] * 1.15 * (1 if r["delta"] >= 0 else -1)
                    ax.annotate(star, (xi, r["delta"] + off), ha="center", fontsize=FS,
                                va="bottom" if r["delta"] >= 0 else "top")
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xticks(range(len(TSTARTS)))
        ax.set_xticklabels([f"t{t}" for t in TSTARTS], fontsize=FS)
        ax.set_xlabel("inversion depth (tstart)", fontsize=FS)
        ax.set_ylabel(f"Δ {name}  (LoRA − no LoRA)", fontsize=FS)
        ax.set_title(f"Paired LoRA effect: {name}", fontsize=FS + 1)
        ax.tick_params(labelsize=FS)
        ax.grid(True, linestyle="--", alpha=0.2)
    axes[1][0].legend(fontsize=FS - 1, loc="lower right", framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / "plots" / f"gen_inputs_h1.{ext}", dpi=150, bbox_inches="tight")

    deltas.to_csv(out / "paired_deltas.csv", index=False)
    pivot = deltas.pivot_table(index=["metric", "tstart"], columns="split",
                               values=["delta", "p"]).round(4)
    (out / "REPORT.md").write_text(
        "# H1: paired LoRA effect, real audio vs generated inputs (cfg_tar 3.5)\n\n"
        "Figure: `plots/gen_inputs_h1.png`. Deltas are LoRA@4000 − no-LoRA, paired over the\n"
        "same 115 rows; CI bars are 95%.\n\n" + pivot.to_markdown() + "\n")
    print(deltas.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
