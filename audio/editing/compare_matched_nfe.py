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

METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir",
           "mulan_dir": "mulan_dir"}
LABELS = {"odeinv": "ODEInv", "ddpm": "DDPM-inv", "sdedit": "SDEdit"}
COLORS = {"ODEInv w/ LoRA bw": "#d62728", "ODEInv (no LoRA)": "#1f77b4",
          "DDPM-inv": "#2ca02c", "SDEdit": "#ff7f0e", "ODEInv w/ LoRA real": "#9467bd",
          "ODEInv w/ LoRA rollout k=4": "#8c564b", "ODEInv w/ LoRA rollout k=8": "#e377c2"}
# Adapter run -> arm label. The trajectory adapter keeps its original label so earlier reports and
# the paper figure read the same; any other adapter is named by its run.
LORA_ARMS = {"saocos_r8_a4_lr5e-5": "ODEInv w/ LoRA bw",
             "saocos_realfndur_r8_a4_lr5e-5": "ODEInv w/ LoRA real",
             "saocos_rollout4_r8_a4_lr5e-5": "ODEInv w/ LoRA rollout k=4",
             "saocos_rollout8_r8_a4_lr5e-5": "ODEInv w/ LoRA rollout k=8"}


def lora_arm(checkpoint: str) -> str:
    """Arm label for a run-name checkpoint field like `saocos_r8_a4_lr5e-5_checkpoint_step_4000`."""
    run = checkpoint.split("_checkpoint_")[0]
    return LORA_ARMS.get(run, f"ODEInv w/ LoRA {run}")


def collect(runs_root: Path, split: str, inversion: str = "lift") -> pd.DataFrame:
    """Read every scored matched-NFE run of one split into one row each.

    Args:
        runs_root: Directory holding the model's run subdirectories.
        split: Benchmark split in the run names (`hparam` or `genhparam`).
        inversion: Which odeinv runs to use: `lift` (ode_invert lifts the clean latent through the
            sigma = 0 step, run names ending `_lift`) or `orig` (the pre-fix runs). DDPM-inv and
            SDEdit do not invert by ODE and are shared by both.

    Returns:
        One row per run, with the arm label, depth, and the mean/SEM of each metric.
    """
    assert inversion in ("lift", "orig"), inversion
    spec = MODELS["stable_audio_nfe"]
    rows = []
    for subdir in spec["subdirs"]:
        for run_dir in sorted((runs_root / subdir).glob(spec["glob"])):
            match = spec["lora"].match(run_dir.name) or spec["base"].match(run_dir.name)
            csv = run_dir / PER_EXAMPLE_CSV
            if match is None or not csv.exists():
                continue
            g = match.groupdict()
            if g.get("split") != split:
                continue
            if not g.get("mode") and (g.get("lift") is not None) != (inversion == "lift"):
                continue  # an odeinv run of the other inversion
            tstart, steps = int(g["tstart"]), int(g["steps"])
            if g.get("checkpoint"):
                arm = lora_arm(g["checkpoint"])
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
    frame = pd.DataFrame(rows).sort_values(["arm", "cfg_tar", "depth"]).reset_index(drop=True)
    if not frame.arm.str.startswith("ODEInv").any():
        raise ValueError(f"no scored odeinv runs with inversion={inversion!r} on split={split}; "
                         "pass --inversion orig for the pre-fix runs")
    return frame


SPLIT_LABEL = {"hparam": "Real audio from MedleyMD", "full": "Real audio from MedleyMD (all 696 edits)",
               "genhparam": "Generated audio from MedleyMD prompts"}
PANELS = [("clap", "Alignment = CLAP"), ("muq", "Alignment = MuQ")]
FS = 16


def lora_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Paired LoRA - no-LoRA delta at matched (tstart, steps, guidance), one block per adapter.

    Pairing must not mix operating points across guidances; only the odeinv arms have adapters.
    """
    key = ["tstart", "steps", "cfg_tar"]
    base = df[df.arm == "ODEInv (no LoRA)"].set_index(key)
    blocks = []
    for arm in sorted(a for a in df.arm.unique() if a.startswith("ODEInv w/ LoRA")):
        lora = df[df.arm == arm].set_index(key)
        shared = lora.index.intersection(base.index)
        blocks.append(pd.DataFrame({
            "arm": arm,
            "depth": lora.loc[shared, "depth"].to_numpy(),
            "cfg_tar": [ix[2] for ix in shared],
            **{m: (lora.loc[shared, m] - base.loc[shared, m]).to_numpy() for m in METRICS},
        }).sort_values(["cfg_tar", "depth"]))
    return pd.concat(blocks, ignore_index=True)


def main(runs_root: str, out_root: str = "output/matched_nfe", split: str = "hparam",
         inversion: str = "lift") -> None:
    """Write the matched-NFE table, the paired LoRA delta, and the front figure.

    Args:
        runs_root: Directory holding `stable_audio/`, e.g. .../edits/medleymd/medleymd.
        out_root: Destination, relative to `audio/`.
        split: `hparam` (real audio), `genhparam` (generated inputs), or `both` for one figure
            with the real front on top and the generated one below, axes shared per column.
        inversion: `lift` for the odeinv runs made with the sigma = 0 lift (`_lift` names), or
            `orig` for the pre-fix ones.
    """
    splits = ["hparam", "genhparam"] if split == "both" else [split]
    frames = {s: collect(Path(runs_root), s, inversion) for s in splits}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = AUDIO_ROOT / out_root / stamp
    out.mkdir(parents=True, exist_ok=True)

    reference = frames[splits[0]]
    budgets = sorted(reference["nfe"].unique())
    cfgs = sorted(reference["cfg_tar"].unique())
    lines = [f"# Stable Audio Open at matched NFE (~{budgets[0]} denoiser calls), "
             f"split={split}\n",
             f"{sum(len(f) for f in frames.values())} runs, {reference['n'].iloc[0]} edits "
             f"each, cfg_tar pooled ({', '.join(f'{c:g}' for c in cfgs)}). "
             f"Points are labelled by inversion depth = tstart/steps. odeinv runs: "
             f"{'sigma = 0 lift (_lift)' if inversion == 'lift' else 'pre-fix inversion'}.\n",
             "Figure: `matched_nfe_front.png`.\n"]

    # One centered title per row, over both panels: a subfigure per split carries it.
    fig = plt.figure(figsize=(6.4 * len(PANELS), 5.9 * len(splits)), layout="constrained")
    fig.suptitle("Audio editing with Stable Audio at matched NFEs",
                 fontsize=19, fontweight="bold")
    subfigs = fig.subfigures(len(splits), 1)
    subfigs = subfigs if isinstance(subfigs, (list, tuple)) or hasattr(subfigs, "__len__") \
        else [subfigs]
    axes = []
    for row, s in enumerate(splits):
        subfigs[row].suptitle(SPLIT_LABEL[s], fontsize=FS + 1)
        axes.append(subfigs[row].subplots(1, len(PANELS)))
    for row, s in enumerate(splits):
        df = frames[s]
        print(f"\n=== split={s}: {len(df)} runs ===")
        show = df[["arm", "cfg_tar", "depth", "tstart", "steps",
                   "lpaps", "clap", "muq", "clap_dir", "mulan_dir"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        delta = lora_delta(df)
        print("\nLoRA - no LoRA, paired at matched depth (LPAPS lower is better):")
        print(delta.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        lines += [f"\n## All runs — {SPLIT_LABEL[s]}\n",
                  show.to_markdown(index=False, floatfmt=".4f"),
                  f"\n## LoRA - no LoRA, paired — {SPLIT_LABEL[s]}\n",
                  delta.to_markdown(index=False, floatfmt="+.4f")]
        df.to_csv(out / f"matched_nfe_runs_{s}.csv", index=False)

        for ax, (metric, name) in zip(axes[row], PANELS):
            for arm, sub in df.groupby("arm"):
                sub = sub.sort_values("lpaps")
                ax.plot(sub["lpaps"], sub[metric], marker="o", ms=7, lw=1.8,
                        color=COLORS.get(arm), label=arm)
            if row == len(splits) - 1:
                ax.set_xlabel("LPAPS to source", fontsize=FS)
            ax.set_ylabel(name, fontsize=FS)
            ax.tick_params(labelsize=FS - 2)
            # The ideal corner: perfectly preserved and perfectly aligned.
            ax.text(0.035, 0.955, "★", transform=ax.transAxes, fontsize=26, color="#f1c40f",
                    ha="center", va="center",
                    path_effects=None)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=FS - 3, loc="lower right")
    for ext in ("png", "svg"):
        fig.savefig(out / f"matched_nfe_front.{ext}", dpi=150, bbox_inches="tight")

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
