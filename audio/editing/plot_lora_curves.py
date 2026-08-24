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

# The other two methods come from the same sweep at the same settings. They are not adapters, so
# they are carried as pseudo-checkpoints purely to put the LoRA differences on a scale where a
# real method difference is visible. Stable Audio has no hparam-grid runs for them -- its DDPM and
# SDEdit baselines exist only at single full-split settings -- so there the no-LoRA front is the
# only reference drawn.
METHOD_LABELS = {"ddpm": "DDPM-inv", "sdedit": "SDEdit"}

# Per model: which subdirectories hold the runs, and how a run directory name parses. The no-LoRA
# pattern must expose tstart and cfg_tar; `mode` is optional and defaults to ddim.
MODELS = {
    "audioldm2": {
        "title": "AudioLDM2",
        "subdirs": ["audioldm2_ddim", "audioldm2_ddpm", "audioldm2_sdedit"],
        "glob": "audioldm2_*hparam*",
        "base": re.compile(
            r"audioldm2_(?P<mode>ddpm|ddim|sdedit)_hparam"
            r"(?:_cfgsrc(?P<cfg_src>[\d.]+))?_cfgtar(?P<cfg_tar>[\d.]+)"
            r"_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
        ),
        "lora": re.compile(
            r"audioldm2_ddimlora_hparam_(?P<checkpoint>.+?)"
            r"_cfgtar(?P<cfg_tar>[\d.]+)_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
        ),
    },
    "stable_audio": {
        "title": "Stable Audio Open",
        "subdirs": ["stable_audio"],
        "glob": "stableaudio_*hparam*",
        "base": re.compile(
            r"stableaudio_ddim_nolora_hparam_cfgtar(?P<cfg_tar>[\d.]+)"
            r"_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
        ),
        "lora": re.compile(
            r"stableaudio_ddimlora_hparam_(?P<checkpoint>.+?)"
            r"_cfgtar(?P<cfg_tar>[\d.]+)_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
        ),
    },
}

# plot label -> column in the consolidated per-example table
METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir",
           "psnr": "psnr", "ssim": "ssim"}
Y_PANELS = [("clap", "CLAP to target caption"), ("muq", "MuQ-MuLan to target caption"),
            ("clap_dir", "Directional CLAP")]
PAIRED = [("lpaps", "LPAPS", "lower"), ("psnr", "mel PSNR", "higher"),
          ("clap", "CLAP", "higher")]




def collect(root: Path, model: str) -> pd.DataFrame:
    """Every scored hparam DDIM run, no-LoRA and LoRA alike, as one row per run.

    Args:
        root: Directory holding the model's run subdirectories.
        model: Key of `MODELS`, which supplies the layout and the name patterns.

    Returns:
        Rows with checkpoint, tstart, cfg_tar and the mean/SEM of each metric.
    """
    spec = MODELS[model]
    rows = []
    candidates = [
        path for subdir in spec["subdirs"] for path in sorted((root / subdir).glob(spec["glob"]))
    ]
    for run_dir in candidates:
        match = spec["lora"].match(run_dir.name) or spec["base"].match(run_dir.name)
        if match is None or not (run_dir / PER_EXAMPLE_CSV).exists():
            continue
        groups = match.groupdict()
        label = groups.get("checkpoint")
        if label is None:
            mode = groups.get("mode") or "ddim"
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
    model: str = "audioldm2",
    runs_root: str | None = None,
    out_root: str = "output/lora_curves",
    cfg_tars: str | None = None,
):
    """Plot the LoRA arm against its no-LoRA twins and write the paired statistics.

    Args:
        model: Which model's runs to read, `audioldm2` or `stable_audio`.
        runs_root: Directory holding the run subdirectories; defaults to the configured outputs.
            Stable Audio's runs sit one level deeper, under `medleymd/medleymd`.
        out_root: Where the timestamped output directory is created.
        cfg_tars: Guidance values to keep, so the baseline is restricted to what the LoRA arm
            covers. None keeps every value present.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    if runs_root:
        root = Path(runs_root)
    else:
        root = Path(PATH_EDIT_OUTPUTS) / "medleymd"
        if model == "stable_audio":
            root = root / "medleymd"
    frame = collect(root, model)
    if frame.empty:
        raise FileNotFoundError(f"no scored {model} hparam runs under {root}")
    keep = sorted(frame["cfg_tar"].unique()) if cfg_tars is None else [
        float(c) for c in str(cfg_tars).split(",")
    ]
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

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    for axis, (y, title) in zip(axes, Y_PANELS):
        for checkpoint, group in frame.groupby("checkpoint"):
            group = group.sort_values("lpaps")
            axis.errorbar(
                group["lpaps"], group[y],
                xerr=group["lpaps_sem"], yerr=group[f"{y}_sem"],
                capsize=2, color=colours[checkpoint], label=short(checkpoint),
                markeredgecolor="black", markeredgewidth=0.8,
                **style(checkpoint),
            )
        # Optimal editing is the top-left corner: nothing changed that should not have, everything
        # changed that should. Unlabelled -- it marks the direction, it is not a data point.
        axis.plot(0.045, 0.955, marker="*", markersize=22, color="gold",
                  markeredgecolor="#7a6000", markeredgewidth=0.8, transform=axis.transAxes,
                  clip_on=False, zorder=6, label="_nolegend_")
        axis.set_xlabel("LPAPS to source $\\downarrow$", fontsize=15)
        axis.set_ylabel(f"{title} $\\uparrow$", fontsize=15)
        axis.tick_params(labelsize=14)
        axis.grid(True, linestyle="--", alpha=0.2)
    axes[0].legend(fontsize=14, loc="lower right", framealpha=0.9)

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
        f"{MODELS[model]['title']}: inversion LoRA on MedleyMD, DDIM grid "
        f"(n={int(frame['n'].iloc[0])} edits per point)",
        fontsize=16,
    )
    figure.tight_layout()
    plot_path = out_dir / "plots" / f"{stamp}_{model}_lora_tradeoff.png"
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    paired = pd.DataFrame(records)
    paired.to_csv(out_dir / f"{stamp}_{model}_lora_paired.csv", index=False)

    lines = [
        f"# {MODELS[model]['title']}: inversion LoRA vs no LoRA -- MedleyMD hparam split, DDIM only",
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
    (out_dir / f"{stamp}_{model}_lora_paired.md").write_text("\n".join(lines) + "\n")

    with (out_dir / "run_meta.json").open("w") as handle:
        json.dump({"timestamp": stamp, "model": model, "runs_root": str(root), "cfg_tars": keep,
                   "checkpoints": checkpoints, "num_runs": len(frame),
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      text=True).strip()}, handle, indent=2)
    print(f"\nwrote {out_dir}\n  {plot_path}\n  {out_dir / f'{stamp}_{model}_lora_paired.md'}")

    print("\nmean paired delta per checkpoint (negative LPAPS = better preservation):")
    summary = paired.pivot_table(index="checkpoint", columns="metric", values="delta", aggfunc="mean")
    print(summary.to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    fire.Fire(main)
