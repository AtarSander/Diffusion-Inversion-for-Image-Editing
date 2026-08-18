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

METRICS = {
    "lpaps": ("lpaps_to_source.csv", "lpaps"),
    "clap": ("clap_to_target_prompt.csv", "clap"),
    "muq": ("mulan_to_target_prompt.csv", "muqt_sim_p0"),
    "clap_dir": ("directional_to_prompts.csv", "clap_dir"),
    "psnr": ("psnr_ssim_per_file.csv", "psnr"),
    "ssim": ("psnr_ssim_per_file.csv", "ssim"),
}
Y_PANELS = [("clap", "CLAP to target caption"), ("muq", "MuQ-MuLan to target caption"),
            ("clap_dir", "Directional CLAP")]
PAIRED = [("lpaps", "LPAPS", "lower"), ("psnr", "mel PSNR", "higher"),
          ("clap", "CLAP", "higher")]


def per_example(run_dir: Path, metric: str) -> pd.Series:
    """One metric's per-example values, indexed by row so runs can be paired."""
    filename, column = METRICS[metric]
    series = pd.read_csv(run_dir / filename, index_col=0)[column]
    series.index = [
        int(str(i).removeprefix("a").removesuffix(".wav")) if str(i).startswith("a") else int(i)
        for i in series.index
    ]
    return series.sort_index()


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
        if match is None or not (run_dir / "lpaps_to_source.csv").exists():
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
        for metric in METRICS:
            if not (run_dir / METRICS[metric][0]).exists():
                continue
            values = per_example(run_dir, metric)
            row["n"] = len(values)
            row[metric] = values.mean()
            row[f"{metric}_sem"] = values.sem()
        rows.append(row)
    return pd.DataFrame(rows)


def short(checkpoint: str) -> str:
    """Compact legend label: training run plus step, EMA marked."""
    if checkpoint == "no LoRA":
        return checkpoint
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

    # Panel row 1: the fronts overlaid. Row 2: paired differences against the no-LoRA twin.
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    for axis, (y, title) in zip(axes[0], Y_PANELS):
        for checkpoint, group in frame.groupby("checkpoint"):
            group = group.sort_values("lpaps")
            axis.errorbar(
                group["lpaps"], group[y],
                xerr=group["lpaps_sem"], yerr=group[f"{y}_sem"],
                capsize=2, color=colours[checkpoint], label=short(checkpoint),
                **style(checkpoint),
            )
        axis.set_xlabel("LPAPS to source $\\downarrow$")
        axis.set_ylabel(f"{title} $\\uparrow$")
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0][0].legend(fontsize=7, loc="lower right")

    baseline = frame[frame["checkpoint"] == "no LoRA"].set_index(["tstart", "cfg_tar"])
    records = []
    for axis, (metric, label, direction) in zip(axes[1], PAIRED):
        for checkpoint in checkpoints + methods:
            xs, deltas, errors = [], [], []
            for _, row in frame[frame["checkpoint"] == checkpoint].sort_values("tstart").iterrows():
                twin = baseline.loc[(row["tstart"], row["cfg_tar"])]
                lora_values = per_example(row["run_dir"], metric)
                base_values = per_example(twin["run_dir"], metric)
                common = lora_values.index.intersection(base_values.index)
                delta = lora_values.loc[common] - base_values.loc[common]
                pvalue = stats.ttest_rel(lora_values.loc[common], base_values.loc[common]).pvalue
                xs.append(row["tstart"] + (row["cfg_tar"] - 9) * 1.5)
                deltas.append(delta.mean())
                errors.append(1.96 * delta.std() / len(delta) ** 0.5)
                records.append({
                    "checkpoint": checkpoint, "tstart": row["tstart"], "cfg_tar": row["cfg_tar"],
                    "metric": metric, "delta": delta.mean(), "ci95": errors[-1],
                    "p": pvalue, "better": direction,
                    "n_better": int((delta < 0).sum() if direction == "lower" else (delta > 0).sum()),
                    "n": len(delta),
                })
            axis.errorbar(xs, deltas, yerr=errors, capsize=2, color=colours[checkpoint],
                          label=short(checkpoint), **style(checkpoint))
        axis.axhline(0, color="black", linewidth=1.0, zorder=1)
        # SDEdit differs from DDIM by ~0.7 LPAPS and the adapters by ~0.002, so a linear axis
        # would render one of the two invisible. Linear inside the adapters' range, log outside.
        adapter_max = max(
            abs(record["delta"])
            for record in records
            if record["metric"] == metric and record["checkpoint"] in checkpoints
        )
        threshold = max(adapter_max * 2, 1e-4)
        axis.set_yscale("symlog", linthresh=threshold)
        axis.axhspan(-threshold, threshold, color="0.9", zorder=0)
        axis.set_xlabel("tstart (jittered by cfg_tar)")
        axis.set_ylabel(
            f"$\\Delta$ {label} vs no-LoRA DDIM "
            f"({'lower' if direction == 'lower' else 'higher'} is better)"
        )
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[1][0].legend(fontsize=7, loc="upper left", ncol=2)

    figure.suptitle(
        f"Inversion LoRA on the MedleyMD hparam split, n={int(frame['n'].iloc[0])} edits per point"
        "\ntop: LoRA fronts on the no-LoRA DDIM baseline (black), with DDPM-inv and SDEdit at the "
        "same settings for scale"
        "\nbottom: paired difference vs no-LoRA DDIM, 95% CI; linear inside the grey band, log "
        "outside it",
        fontsize=11,
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
