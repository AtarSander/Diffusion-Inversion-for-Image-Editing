# ABOUTME: Builds the matched-NFE comparison table and figure for Stable Audio: every method at
# ABOUTME: one denoiser-call budget, with inversion depth rather than tstart as the front axis.

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

from plot_lora_curves import MODELS  # noqa: E402
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir"}
LABELS = {"odeinv": "ODEInv", "ddpm": "DDPM-inv", "sdedit": "SDEdit"}
COLORS = {"ODEInv w/ LoRA bw": "#d62728", "ODEInv (no LoRA)": "#1f77b4",
          "DDPM-inv": "#2ca02c", "SDEdit": "#ff7f0e"}


def collect(runs_root: Path) -> pd.DataFrame:
    """Read every scored matched-NFE run into one row each.

    Args:
        runs_root: Directory holding the model's run subdirectories.

    Returns:
        One row per run, with the arm label, depth, and the mean/SEM of each metric.
    """
    spec = MODELS["stable_audio_nfe"]
    rows = []
    for subdir in spec["subdirs"]:
        for run_dir in sorted((runs_root / subdir).glob(spec["glob"])):
            match = spec["lora"].match(run_dir.name) or spec["base"].match(run_dir.name)
            csv = run_dir / PER_EXAMPLE_CSV
            if match is None or not csv.exists():
                continue
            g = match.groupdict()
            tstart, steps = int(g["tstart"]), int(g["steps"])
            if g.get("checkpoint"):
                arm = "ODEInv w/ LoRA bw"
            else:
                arm = LABELS[g.get("mode") or "odeinv"]
                if arm == "ODEInv":
                    arm = "ODEInv (no LoRA)"
            frame = pd.read_csv(csv)
            row = {"arm": arm, "tstart": tstart, "steps": steps, "nfe": int(g["nfe"]),
                   # tstart/steps is the fraction of the grid the inversion actually walks, which
                   # is the axis a fixed budget leaves free.
                   "depth": round(100 * tstart / steps), "n": len(frame),
                   "cfg_tar": float(g["cfg_tar"]), "run": run_dir.name}
            for name, column in METRICS.items():
                row[name] = frame[column].mean()
                row[f"{name}_sem"] = frame[column].sem()
            rows.append(row)
    assert rows, f"no scored matched-NFE runs under {runs_root}"
    return pd.DataFrame(rows).sort_values(["arm", "cfg_tar", "depth"]).reset_index(drop=True)


def main(runs_root: str, out_root: str = "output/matched_nfe") -> None:
    """Write the matched-NFE table, the paired LoRA delta, and the front figure.

    Args:
        runs_root: Directory holding `stable_audio/`, e.g. .../edits/medleymd/medleymd.
        out_root: Destination, relative to `audio/`.
    """
    df = collect(Path(runs_root))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = AUDIO_ROOT / out_root / stamp
    out.mkdir(parents=True, exist_ok=True)

    budgets = sorted(df["nfe"].unique())
    cfgs = sorted(df["cfg_tar"].unique())
    print(f"{len(df)} runs, NFE {budgets}, cfg_tar {cfgs}, n={df['n'].iloc[0]} edits each\n")
    show = df[["arm", "cfg_tar", "depth", "tstart", "steps", "lpaps", "clap", "muq", "clap_dir"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Paired LoRA delta at matched (tstart, steps, guidance): the only arm with an adapter is
    # odeinv, and pairing must not mix operating points across guidances.
    lora = df[df.arm == "ODEInv w/ LoRA bw"].set_index(["tstart", "steps", "cfg_tar"])
    base = df[df.arm == "ODEInv (no LoRA)"].set_index(["tstart", "steps", "cfg_tar"])
    shared = lora.index.intersection(base.index)
    delta = pd.DataFrame({
        "depth": lora.loc[shared, "depth"].to_numpy(),
        "cfg_tar": [ix[2] for ix in shared],
        **{m: (lora.loc[shared, m] - base.loc[shared, m]).to_numpy() for m in METRICS},
    }).sort_values(["cfg_tar", "depth"])
    print("\nLoRA - no LoRA, paired at matched depth (LPAPS lower is better):")
    print(delta.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.suptitle(
        f"Stable Audio Open at a matched budget of ~{budgets[0]} denoiser calls — MedleyMD hparam "
        f"split, {df['n'].iloc[0]} edits, cfg_tar pooled: {', '.join(f'{c:g}' for c in cfgs)} "
        "— points labelled depth/w",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for ax, (metric, name) in zip(axes, [("clap", "CLAP to target caption"),
                                         ("muq", "MuQ-MuLan to target"),
                                         ("clap_dir", "Directional CLAP")]):
        for arm, sub in df.groupby("arm"):
            sub = sub.sort_values("lpaps")
            ax.errorbar(sub["lpaps"], sub[metric], xerr=sub["lpaps_sem"], yerr=sub[f"{metric}_sem"],
                        marker="o", ms=6, lw=1.6, capsize=2.5, color=COLORS.get(arm), label=arm)
            for _, r in sub.iterrows():
                label = f"{r['depth']}%" if len(cfgs) == 1 else f"{r['depth']}%/{r['cfg_tar']:g}"
                ax.annotate(label, (r["lpaps"], r[metric]), fontsize=7,
                            textcoords="offset points", xytext=(4, 4), color=COLORS.get(arm))
        ax.set(xlabel="LPAPS to source (lower = better preserved)", ylabel=name, title=name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"matched_nfe_front.{ext}", dpi=150, bbox_inches="tight")

    df.to_csv(out / "matched_nfe_runs.csv", index=False)
    lines = [f"# Stable Audio Open at matched NFE (~{budgets[0]} denoiser calls)\n",
             f"{len(df)} runs, {df['n'].iloc[0]} edits each, cfg_tar pooled "
             f"({', '.join(f'{c:g}' for c in cfgs)}). "
             f"Points are labelled by inversion depth = tstart/steps.\n",
             "Figure: `matched_nfe_front.png`.\n", "## All runs\n",
             show.to_markdown(index=False, floatfmt=".4f"), "\n## LoRA - no LoRA, paired\n",
             delta.to_markdown(index=False, floatfmt="+.4f"), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
