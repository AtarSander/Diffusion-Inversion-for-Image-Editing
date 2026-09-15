# ABOUTME: The guided-inversion comparison restricted to arms whose deployment actually differs:
# ABOUTME: unguided vs guided inversion, and the two adapters that only guided inversion invokes.

import re
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
CSV = "per_example_metrics.csv"
TAIL = r"_cfgtar(?P<tar>[\d.]+)_t(?P<t>\d+)_s100$"

# Only these four. The cfg_src=1.0 adapter arms are excluded on purpose: guided() returns after
# the conditional call at scale 1.0, so an adapter's unconditional branch is never invoked there
# and those arms differ from each other only in how the conditional adapter was trained.
ARMS = [
    ("ODE Inv. (cfgs = 1)", r"stableaudio_odeinv_nolora_hparam" + TAIL, "#7f7f7f", "--", "s"),
    ("ODE Inv. (cfgs = 3.5)", r"stableaudio_odeinv_nolora_hparam_cfgsrc3\.5" + TAIL,
     "#d62728", "--", "s"),
    ("LoRA Inv. (shared_cfg, cfgs = 3.5)",
     r"stableaudio_odeinvlora_hparam_saocos_cfg35_r8_a4_lr5e-5_checkpoint_step_2000_cfgsrc3\.5"
     + TAIL, "#8c564b", "-", "o"),
    ("LoRA Inv. (pair_branch, cfgs = 3.5)",
     r"stableaudio_odeinvlora_hparam_saocos_cfg35pair_r8_a4_lr5e-5_checkpoint_step_2000_cfgsrc3\.5"
     + TAIL, "#e377c2", "-", "o"),
]
METRICS = [("clap", "CLAP to target caption"), ("muq", "MuQ-MuLan to target")]


def collect(root: Path) -> pd.DataFrame:
    """One row per (arm, cfg_tar, tstart) from the scored per-example tables."""
    rows = []
    for label, pattern, _, _, _ in ARMS:
        head = re.compile(pattern)
        for run in sorted(root.glob("stableaudio_*_s100")):
            m = head.match(run.name)
            if not m or not (run / CSV).exists():
                continue
            f = pd.read_csv(run / CSV)
            rows.append({"arm": label, "cfg_tar": float(m.group("tar")),
                         "tstart": int(m.group("t")), "n": len(f),
                         "lpaps": f.lpaps.mean(), "lpaps_sem": f.lpaps.sem(),
                         "clap": f.clap.mean(), "clap_sem": f.clap.sem(),
                         "muq": f.muqt_sim_p0.mean(), "muq_sem": f.muqt_sim_p0.sem()})
    df = pd.DataFrame(rows)
    assert not df.empty, f"no scored runs under {root}"
    assert not df.duplicated(["arm", "cfg_tar", "tstart"]).any(), "two runs in one cell"
    return df


def main(runs_root: str, cfg_tar: float = 3.5, out_root: str = "output/guided_final") -> None:
    """Draw CLAP and MuQ fronts for the four comparable arms.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        cfg_tar: Target guidance to show; the arms are compared at one value at a time so a
            point's position is not confounded by which guidance produced it.
        out_root: Destination, relative to `audio/`.
    """
    df = collect(Path(runs_root))
    df = df[df.cfg_tar == cfg_tar]
    assert not df.empty, f"nothing scored at cfg_tar={cfg_tar}"
    print(df.sort_values(["arm", "tstart"])[["arm", "tstart", "n", "lpaps", "clap", "muq"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4))
    fig.suptitle(
        f"Stable Audio Open — guided vs unguided ODE inversion "
        f"(cfg_tar = {cfg_tar:g}, 100-step grid, {int(df.n.iloc[0])} edits, points labelled tstart)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ax, (metric, name) in zip(axes, METRICS):
        for label, _, color, style, marker in ARMS:
            sub = df[df.arm == label].sort_values("tstart")
            if sub.empty:
                continue
            ax.errorbar(sub["lpaps"], sub[metric], xerr=sub["lpaps_sem"],
                        yerr=sub[f"{metric}_sem"], fmt=marker + style, ms=6, lw=1.8,
                        capsize=2.5, color=color, label=label, alpha=0.9)
            for _, r in sub.iterrows():
                ax.annotate(f"{int(r['tstart'])}", (r["lpaps"], r[metric]), fontsize=7.5,
                            textcoords="offset points", xytext=(5, 4), color=color)
        ax.set(xlabel="LPAPS to source (lower = better preserved)", ylabel=name, title=name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"guided_final_cfgtar{cfg_tar:g}"
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / "REPORT.md").write_text(
        f"# Guided vs unguided ODE inversion, cfg_tar = {cfg_tar:g}\n\n"
        "Only arms whose deployment differs. The cfg_src=1.0 adapter arms are excluded: "
        "`guided()` returns after the conditional call at scale 1.0, so an adapter's "
        "unconditional branch is never invoked there.\n\n"
        + df.sort_values(["arm", "tstart"]).to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
