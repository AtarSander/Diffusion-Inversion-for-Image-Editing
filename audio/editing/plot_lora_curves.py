# ABOUTME: Compare the inversion-LoRA arm against its no-LoRA twins on the hparam sweep, as an
# ABOUTME: overlay of the tradeoff fronts plus the paired per-setting differences that decide it.

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from editing.AudioEditingCode.code.env import PATH_EDIT_OUTPUTS  # noqa: E402
from editing.run_metrics import PER_EXAMPLE_CSV, per_example  # noqa: E402

BASE_RE = re.compile(
    r"audioldm2_(?P<mode>ddpm|ddim|sdedit)_hparam"
    r"(?:_cfgsrc(?P<cfg_src>[\d.]+))?_cfgtar(?P<cfg_tar>[\d.]+)"
    r"_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
)

# The other two methods come from the same sweep at the same settings. They are not adapters, so
# they are carried as pseudo-checkpoints purely to put the LoRA differences on a scale where a
# real method difference is visible.
METHOD_LABELS = {"ddpm": "DDPM-inv", "sdedit": "SDEdit"}
LORA_RE = re.compile(
    r"audioldm2_ddimlora_hparam_(?P<checkpoint>.+?)"
    r"_cfgtar(?P<cfg_tar>[\d.]+)_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
)

# plot label -> column in the consolidated per-example table
METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir",
           "psnr": "psnr", "ssim": "ssim"}
Y_PANELS = [("clap", "CLAP to target caption"), ("muq", "MuQ-MuLan to target caption"),
            ("clap_dir", "Directional CLAP")]
PAIRED = [("lpaps", "LPAPS", "lower"), ("psnr", "mel PSNR", "higher"),
          ("clap", "CLAP", "higher")]




def collect(root: Path) -> pd.DataFrame:
    """Every scored hparam DDIM run, no-LoRA and LoRA alike, as one row per run.

    Args:
        root: Directory holding `audioldm2_ddim/<run>/`.

    Returns:
        Rows with checkpoint, tstart, cfg_tar and the mean/SEM of each metric.
    """
    rows = []
    candidates = [
        path
        for mode in ("ddim", "ddpm", "sdedit")
        for path in sorted((root / f"audioldm2_{mode}").glob(f"audioldm2_{mode}*hparam*"))
    ]
    for run_dir in candidates:
        match = LORA_RE.match(run_dir.name) or BASE_RE.match(run_dir.name)
        if match is None or not (run_dir / PER_EXAMPLE_CSV).exists():
            continue
        groups = match.groupdict()
        label = groups.get("checkpoint")
        if label is None:
            mode = groups["mode"]
            label = "no LoRA" if mode == "ddim" else METHOD_LABELS[mode]
        row = {
            "checkpoint": label,
            "tstart": int(groups["tstart"]),
            "cfg_tar": float(groups["cfg_tar"]),
            "run_dir": run_dir,
        }
        for metric, column in METRICS.items():
            try:
                values = per_example(run_dir, column)
            except (FileNotFoundError, KeyError):
                continue
            row["n"] = len(values)
            row[metric] = values.mean()
            row[f"{metric}_sem"] = values.sem()
        rows.append(row)
    return pd.DataFrame(rows)


def short(checkpoint: str) -> str:
    """Compact legend label: training run plus step, EMA marked."""
    if checkpoint == "no LoRA":
        return "DDIM Inv. (no LoRA)"
    stem = checkpoint.replace("_checkpoint_step_", " @")
    return stem.replace("_ema", " EMA")


def main(
    runs_root: str | None = None,
    out_root: str = "output/lora_curves",
    cfg_tars: str = "6.0,12.0",
):
    """Plot the LoRA arm against its no-LoRA twins and write the paired statistics.

    Args:
        runs_root: Directory holding `audioldm2_ddim/<run>/`; defaults to the configured outputs.
        out_root: Where the timestamped output directory is created.
        cfg_tars: Guidance values the LoRA arm covers, so the baseline is restricted to match.
    """
    root = Path(runs_root) if runs_root else Path(PATH_EDIT_OUTPUTS) / "medleymd"
    frame = collect(root)
    keep = [float(c) for c in str(cfg_tars).split(",")]
    frame = frame[frame["cfg_tar"].isin(keep)].copy()
    methods = [c for c in METHOD_LABELS.values() if c in set(frame["checkpoint"])]
    checkpoints = [
        c for c in frame["checkpoint"].unique() if c != "no LoRA" and c not in methods
    ]
    print(f"{len(frame)} runs: 1 baseline + {len(checkpoints)} checkpoints "
          f"+ {len(methods)} methods, "
          f"tstart {sorted(frame['tstart'].unique())}, cfg_tar {sorted(frame['cfg_tar'].unique())}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / stamp
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    palette = plt.get_cmap("tab10")
    colours = {"no LoRA": "black", "DDPM-inv": "#1f77b4", "SDEdit": "#2ca02c"}
    for i, checkpoint in enumerate(checkpoints):
        colours[checkpoint] = palette(i % 10)

    def style(name):
        """Reference methods are drawn heavy and dashed; adapters thin, so overlap stays legible."""
        if name == "no LoRA":
            return {"marker": "o", "markersize": 6, "linewidth": 2.4, "alpha": 1.0, "zorder": 4}
        if name in methods:
            return {"marker": "^", "markersize": 6, "linewidth": 2.0, "alpha": 0.9,
                    "linestyle": "--", "zorder": 3}
        return {"marker": "s", "markersize": 4, "linewidth": 1.0, "alpha": 0.75, "zorder": 2}

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for axis, (y, title) in zip(axes, Y_PANELS):
        for checkpoint, group in frame.groupby("checkpoint"):
            group = group.sort_values("lpaps")
            axis.errorbar(
                group["lpaps"], group[y],
                xerr=group["lpaps_sem"], yerr=group[f"{y}_sem"],
                capsize=2, color=colours[checkpoint], label=short(checkpoint),
                **style(checkpoint),
            )
        # Optimal editing is the top-left corner: nothing changed that should not have, everything
        # changed that should. Unlabelled -- it marks the direction, it is not a data point.
        axis.plot(0.045, 0.955, marker="*", markersize=22, color="gold",
                  markeredgecolor="#7a6000", markeredgewidth=0.8, transform=axis.transAxes,
                  clip_on=False, zorder=6, label="_nolegend_")
        axis.set_xlabel("LPAPS to source $\\downarrow$")
        axis.set_ylabel(f"{title} $\\uparrow$")
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(fontsize=7, loc="lower right")

    # The paired differences are no longer plotted, but they are what the markdown reports: an
    # effect of ~0.002 LPAPS against a spread ten times larger is only resolvable pairwise.
    baseline = frame[frame["checkpoint"] == "no LoRA"].set_index(["tstart", "cfg_tar"])
    records = []
    for metric, label, direction in PAIRED:
        for checkpoint in checkpoints + methods:
            for _, row in frame[frame["checkpoint"] == checkpoint].sort_values("tstart").iterrows():
                twin = baseline.loc[(row["tstart"], row["cfg_tar"])]
                lora_values = per_example(row["run_dir"], METRICS[metric])
                base_values = per_example(twin["run_dir"], METRICS[metric])
                common = lora_values.index.intersection(base_values.index)
                delta = lora_values.loc[common] - base_values.loc[common]
                pvalue = stats.ttest_rel(lora_values.loc[common], base_values.loc[common]).pvalue
                records.append({
                    "checkpoint": checkpoint, "tstart": row["tstart"], "cfg_tar": row["cfg_tar"],
                    "metric": metric, "delta": delta.mean(),
                    "ci95": 1.96 * delta.std() / len(delta) ** 0.5,
                    "p": pvalue, "better": direction,
                    "n_better": int((delta < 0).sum() if direction == "lower" else (delta > 0).sum()),
                    "n": len(delta),
                })

    figure.suptitle(
        f"Inversion LoRA on MedleyMD, DDIM grid (n={int(frame['n'].iloc[0])} edits per point)",
        fontsize=12,
    )
    figure.tight_layout()
    plot_path = out_dir / "plots" / f"{stamp}_lora_tradeoff.png"
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    paired = pd.DataFrame(records)
    paired.to_csv(out_dir / f"{stamp}_lora_paired.csv", index=False)

    lines = [
        f"# Inversion LoRA vs no LoRA -- MedleyMD hparam split, DDIM only",
        "",
        f"Generated {stamp} from `{root}` at commit "
        f"`{subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}`.",
        f"{len(frame)} runs, {int(frame['n'].iloc[0])} edits each. Every LoRA run is paired with "
        "the no-LoRA run at identical tstart and cfg_tar over the same rows.",
        "",
        f"Figure: `{plot_path}`",
        "",
        "## Paired differences, summarised per checkpoint",
        "",
        "| checkpoint | metric | mean delta | worst-case CI | settings significant | settings better |",
        "|---|---|---|---|---|---|",
    ]
    for checkpoint in checkpoints:
        for metric, label, direction in PAIRED:
            block = paired[(paired["checkpoint"] == checkpoint) & (paired["metric"] == metric)]
            significant = int((block["p"] < 0.05).sum())
            improved = int(
                ((block["delta"] < 0) if direction == "lower" else (block["delta"] > 0)).sum()
            )
            widest = block.loc[block["ci95"].idxmax()]
            lines.append(
                f"| {short(checkpoint)} | {label} | {block['delta'].mean():+.4f} | "
                f"±{widest['ci95']:.4f} | {significant}/{len(block)} | {improved}/{len(block)} |"
            )
    lines += ["", "## Every setting", "",
              "| checkpoint | tstart | cfg_tar | metric | delta | 95% CI | p | rows better |",
              "|---|---|---|---|---|---|---|---|"]
    for _, row in paired.iterrows():
        lines.append(
            f"| {short(row['checkpoint'])} | {row['tstart']} | {row['cfg_tar']:g} | {row['metric']} | "
            f"{row['delta']:+.4f} | ±{row['ci95']:.4f} | {row['p']:.2e} | "
            f"{row['n_better']}/{row['n']} |"
        )
    (out_dir / f"{stamp}_lora_paired.md").write_text("\n".join(lines) + "\n")

    with (out_dir / "run_meta.json").open("w") as handle:
        json.dump({"timestamp": stamp, "runs_root": str(root), "cfg_tars": keep,
                   "checkpoints": checkpoints, "num_runs": len(frame),
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      text=True).strip()}, handle, indent=2)
    print(f"\nwrote {out_dir}\n  {plot_path}\n  {out_dir / f'{stamp}_lora_paired.md'}")

    print("\nmean paired delta per checkpoint (negative LPAPS = better preservation):")
    summary = paired.pivot_table(index="checkpoint", columns="metric", values="delta", aggfunc="mean")
    print(summary.to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    fire.Fire(main)
