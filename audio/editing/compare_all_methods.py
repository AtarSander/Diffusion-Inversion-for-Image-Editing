# ABOUTME: Every Stable Audio method on one front, pooling all cfg_tar values per arm on the
# ABOUTME: 100-step grid: DDPM-inv, SDEdit, ODEInv with and without each adapter variant.

import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))
sys.path.insert(0, str(AUDIO_ROOT / "editing"))
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir",
           "mulan_dir": "mulan_dir"}

# name -> (head pattern with a {split} slot, colour, style). cfg_tar is captured, not pinned:
# every guidance the arm was run at lands on its curve, since guidance is a per-method operating
# knob, not a separate method. The rejected ddim sampler is deliberately absent: it runs the DiT
# off-schedule (output/sao_schedules/REPORT.md) and belongs to a different comparison.
ARMS = {
    "DDPM-inv":            (r"stableaudio_ddpm_{split}", "#2ca02c", "-"),
    "SDEdit":              (r"stableaudio_sdedit_{split}", "#ff7f0e", "-"),
    "ODEInv (no LoRA)":    (r"stableaudio_odeinv_nolora_{split}", "#7f7f7f", "--"),
    "ODEInv + LoRA @2000": (r"stableaudio_odeinvlora_{split}_saocos_r8_a4_lr5e-5"
                            r"_checkpoint_step_2000", "#1f77b4", "-"),
    "ODEInv + LoRA @4000": (r"stableaudio_odeinvlora_{split}_saocos_r8_a4_lr5e-5"
                            r"_checkpoint_step_4000", "#9467bd", "-"),
    "ODEInv + cycle k=2":  (r"stableaudio_odeinvlora_{split}_saocos_cyc_k2_g1\.0_r8_a4_lr5e-5"
                            r"_checkpoint_step_2000", "#d62728", "-"),
    "ODEInv + cycle k=3":  (r"stableaudio_odeinvlora_{split}_saocos_cyc_k3_g1\.0_r8_a4_lr5e-5"
                            r"_checkpoint_step_2000", "#e377c2", "-"),
}
TAIL = r"_cfgtar(?P<cfg>[\d.]+)_t(?P<tstart>\d+)_s100$"


def collect(root: Path, split: str) -> pd.DataFrame:
    """One row per (arm, cfg_tar, tstart), from the scored per-example tables."""
    rows = []
    for arm, (pattern, _, _) in ARMS.items():
        head = re.compile(pattern.format(split=split) + TAIL)
        for run in sorted(root.glob("stableaudio_*_s100")):
            match = head.match(run.name)
            if not match or not (run / PER_EXAMPLE_CSV).exists():
                continue
            frame = pd.read_csv(run / PER_EXAMPLE_CSV)
            row = {"arm": arm, "cfg_tar": float(match.group("cfg")),
                   "tstart": int(match.group("tstart")), "n": len(frame)}
            for name, col in METRICS.items():
                row[name] = frame[col].mean()
                row[f"{name}_sem"] = frame[col].sem()
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no scored runs matching split={split!r} under {root}")
    dupes = df.duplicated(["arm", "cfg_tar", "tstart"], keep=False)
    assert not dupes.any(), f"two runs in one cell:\n{df[dupes]}"
    return df.sort_values(["arm", "cfg_tar", "tstart"]).reset_index(drop=True)


def main(runs_root: str, out_root: str = "output/all_methods", split: str = "hparam") -> None:
    """Print the table and draw the front.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
        split: Benchmark split the runs were produced with (names the run directories), e.g.
            `hparam` for the real-audio sweep or `genhparam` for the generated-input one.
    """
    df = collect(Path(runs_root), split)
    print(df[["arm", "cfg_tar", "tstart", "n", "lpaps", "clap", "muq", "clap_dir", "mulan_dir"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cfgs = sorted(df["cfg_tar"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.0))
    fig.suptitle(
        f"Stable Audio Open, every method on the 100-step grid, split={split} "
        f"(cfg_tar pooled: {', '.join(f'{c:g}' for c in cfgs)}) — points labelled tstart/w",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for ax, (metric, name) in zip(axes, [("clap", "CLAP to target caption"),
                                         ("muq", "MuQ-MuLan to target"),
                                         ("clap_dir", "Directional CLAP"),
                                         ("mulan_dir", "Directional MuLan")]):
        for arm, (_, color, style) in ARMS.items():
            sub = df[df.arm == arm].sort_values("lpaps")
            if sub.empty:
                continue
            ax.errorbar(sub["lpaps"], sub[metric], xerr=sub["lpaps_sem"],
                        yerr=sub[f"{metric}_sem"], fmt="o" + style, ms=5.5, lw=1.6, capsize=2.5,
                        color=color, label=arm, alpha=0.9)
            for _, r in sub.iterrows():
                ax.annotate(f"{int(r['tstart'])}/{r['cfg_tar']:g}", (r["lpaps"], r[metric]),
                            fontsize=6.5, textcoords="offset points", xytext=(4, 4), color=color)
        ax.set(xlabel="LPAPS to source (lower = better preserved)", ylabel=name, title=name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"all_methods.{ext}", dpi=150, bbox_inches="tight")
    df.to_csv(out / "all_methods.csv", index=False)
    (out / "REPORT.md").write_text(
        f"# Every Stable Audio method, 100-step grid, split={split}, "
        f"cfg_tar pooled ({', '.join(f'{c:g}' for c in cfgs)})\n\n"
        "Figure: `all_methods.png`. Points are labelled tstart/cfg_tar.\n\n"
        + df[["arm", "cfg_tar", "tstart", "n", "lpaps", "clap", "muq", "clap_dir", "mulan_dir"]]
        .to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
