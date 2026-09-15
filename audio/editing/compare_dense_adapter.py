# ABOUTME: The H2 verdict figure: paired effects of the coarse- and dense-trained adapters
# ABOUTME: against their shared no-LoRA twin, per (cfg_tar, tstart) cell — identical bars = null.

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

NOLORA = "stableaudio_odeinv_nolora_hparam_cfgtar{c}_t{t}_s100"
ARMS = {
    "coarse (100-step trajectories)": (
        "stableaudio_odeinvlora_hparam_saocos_r8_a4_lr5e-5_checkpoint_step_4000"
        "_cfgtar{c}_t{t}_s100", "#4c72b0"),
    "dense (991-step trajectories)": (
        "stableaudio_odeinvlora_hparam_saocos_dense991_r8_a4_lr5e-5_checkpoint_step_3000"
        "_cfgtar{c}_t{t}_s100", "#c44e52"),
}
CELLS = [(c, t) for c in ["3.5", "7.0"] for t in [25, 50, 75, 99]]
PANELS = [("lpaps", "Δ LPAPS (lower = better)"), ("clap", "Δ CLAP"),
          ("muqt_sim_p0", "Δ MuQ")]
FS = 14


def main(runs_root: str, out_root: str = "output/dense_pairs") -> None:
    """Draw the paired-effect comparison of the two adapters.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
    """
    root = Path(runs_root)

    def frame(pattern: str, c: str, t: int) -> pd.DataFrame:
        df = pd.read_csv(root / pattern.format(c=c, t=t) / PER_EXAMPLE_CSV)
        assert len(df) == 115
        return df.sort_values("position").reset_index(drop=True)

    rows = []
    for c, t in CELLS:
        base = frame(NOLORA, c, t)
        for arm, (pattern, _) in ARMS.items():
            lora = frame(pattern, c, t)
            assert (base["position"] == lora["position"]).all()
            for metric, _ in PANELS:
                d = lora[metric] - base[metric]
                rows.append({"arm": arm, "cell": f"w{c}\nt{t}", "metric": metric,
                             "delta": d.mean(),
                             "ci95": d.sem() * stats.t.ppf(0.975, len(d) - 1)})
    deltas = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.4 * len(PANELS), 5.4))
    fig.suptitle("Trajectory sampling density does not change the adapter "
                 "(paired effect vs the shared no-LoRA twin, 115 edits/cell)",
                 fontsize=FS + 3, fontweight="bold")
    cells = [f"w{c}\nt{t}" for c, t in CELLS]
    width = 0.38
    for ax, (metric, label) in zip(axes, PANELS):
        for k, (arm, (_, color)) in enumerate(ARMS.items()):
            sub = deltas[(deltas.arm == arm) & (deltas.metric == metric)]
            sub = sub.set_index("cell").loc[cells]
            x = [i + (k - 0.5) * width for i in range(len(cells))]
            ax.bar(x, sub["delta"], width, yerr=sub["ci95"], capsize=3,
                   color=color, edgecolor="black", linewidth=0.8, label=arm)
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, fontsize=FS - 1)
        ax.set_ylabel(label, fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        ax.grid(True, linestyle="--", alpha=0.2)
    axes[0].legend(fontsize=FS - 1, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / "plots" / f"dense_vs_coarse.{ext}", dpi=150, bbox_inches="tight")
    deltas.to_csv(out / "paired_deltas.csv", index=False)
    (out / "REPORT.md").write_text(
        "# H2: coarse- vs dense-trained adapter, paired vs the shared no-LoRA twin\n\n"
        "Figure: `plots/dense_vs_coarse.png`. Bars are per-cell mean paired deltas, 95% CI.\n\n"
        + deltas.to_markdown(index=False, floatfmt="+.4f") + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
