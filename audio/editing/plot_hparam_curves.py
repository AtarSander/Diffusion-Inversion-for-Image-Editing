# ABOUTME: Plot the preservation/adherence tradeoff curves for DDPM-inv, DDIM-inv and SDEdit from
# ABOUTME: the scored hparam sweep, with a markdown mirror of every number in the figure.

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from editing.AudioEditingCode.code.env import PATH_EDIT_OUTPUTS  # noqa: E402
from editing.run_metrics import PER_EXAMPLE_CSV, per_example  # noqa: E402

RUN_RE = re.compile(
    r"audioldm2_(?P<mode>ddpm|ddim|sdedit)"
    r"(?:_(?P<split>hparam|test|loc))?"
    r"(?:_cfgsrc(?P<cfg_src>[\d.]+))?"
    r"_cfgtar(?P<cfg_tar>[\d.]+)_t(?P<tstart>\d+)_s(?P<steps>\d+)$"
)

# plot label -> column in the consolidated per-example table. LPAPS is the x axis.
METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir",
           "muq_dir": "mulan_dir", "psnr": "psnr", "ssim": "ssim"}

MODE_LABELS = {"ddpm": "DDPM-inv", "ddim": "DDIM-inv", "sdedit": "SDEdit"}
MODE_COLORS = {"ddpm": "#1f77b4", "ddim": "#d62728", "sdedit": "#2ca02c"}
Y_PANELS = [("clap", "CLAP to target caption"), ("muq", "MuQ-MuLan to target caption"),
            ("clap_dir", "Directional CLAP")]


def load_run(run_dir: Path) -> dict | None:
    """Mean and SEM of every available metric for one scored run.

    Args:
        run_dir: Run directory holding the per-example CSVs (the parent of `audios/`).

    Returns:
        A row dict, or None if the run has no LPAPS yet, i.e. it was never scored.
    """
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return None
    if not (run_dir / PER_EXAMPLE_CSV).exists():
        return None

    row = {
        "mode": match["mode"],
        "tstart": int(match["tstart"]),
        "cfg_tar": float(match["cfg_tar"]),
        "cfg_src": float(match["cfg_src"]) if match["cfg_src"] else None,
        "steps": int(match["steps"]),
        "run": run_dir.name,
    }
    for label, column in METRICS.items():
        try:
            series = per_example(run_dir, column)
        except (FileNotFoundError, KeyError):
            row[label] = row[f"{label}_sem"] = float("nan")
            continue
        row["n"] = len(series)
        row[label] = series.mean()
        row[f"{label}_sem"] = series.sem()
    return row


def pareto_front(frame: pd.DataFrame, y: str) -> pd.DataFrame:
    """Rows not dominated on (low LPAPS, high y), sorted by LPAPS.

    A point is dominated when another point of the same method is at least as good on both axes,
    which is the only honest way to compare methods whose knobs trade the two off.

    Args:
        frame: Rows for one method.
        y: Column that is better when larger.

    Returns:
        The non-dominated rows, ascending in LPAPS.
    """
    ordered = frame.sort_values("lpaps")
    keep, best = [], -float("inf")
    for _, row in ordered.iterrows():
        if row[y] > best:
            keep.append(row)
            best = row[y]
    return pd.DataFrame(keep)


def main(
    runs_root: str | None = None,
    pattern: str = "audioldm2_*_hparam_*",
    out_root: str = "output/hparam_curves",
    split: str = "hparam",
):
    """Collect the scored sweep runs, plot the tradeoff curves and write the markdown mirror.

    Args:
        runs_root: Directory holding `audioldm2_<mode>/<run>/`. Defaults to the configured edit
            outputs, so pointing at a mounted server checkout is a one-flag change.
        pattern: Glob for the run directories to include.
        out_root: Where the timestamped output directory is created.
        split: Recorded in the metadata; the row count is read from the CSVs, not assumed.
    """
    root = Path(runs_root) if runs_root else Path(PATH_EDIT_OUTPUTS) / "medleymd"
    rows = [
        row
        for mode_dir in sorted(root.glob("audioldm2_*"))
        for run_dir in sorted(mode_dir.glob(pattern))
        if (row := load_run(run_dir)) is not None
    ]
    if not rows:
        raise FileNotFoundError(f"No scored runs matching {pattern!r} under {root}")
    frame = pd.DataFrame(rows).sort_values(["mode", "tstart", "cfg_tar"])
    print(f"{len(frame)} scored runs from {root}")
    print(frame.groupby("mode").size().to_string())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / stamp
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(Y_PANELS), figsize=(5.2 * len(Y_PANELS), 4.6))
    for axis, (y, title) in zip(axes, Y_PANELS):
        for mode, group in frame.groupby("mode"):
            colour = MODE_COLORS[mode]
            axis.scatter(group["lpaps"], group[y], s=18, alpha=0.35, color=colour)
            front = pareto_front(group, y)
            axis.errorbar(
                front["lpaps"], front[y],
                xerr=front["lpaps_sem"], yerr=front[f"{y}_sem"],
                marker="o", markersize=5, linewidth=1.8, capsize=2,
                color=colour, label=f"{MODE_LABELS[mode]} ({len(front)}/{len(group)} on front)",
            )
        axis.set_xlabel("LPAPS to source $\\downarrow$")
        axis.set_ylabel(f"{title} $\\uparrow$")
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        f"AudioLDM2 editing tradeoff, MedleyMD {split} split "
        f"(n={int(frame['n'].iloc[0])} edits per point, {len(frame)} configs)\n"
        "faint points are dominated settings; lines join each method's Pareto front",
        fontsize=10,
    )
    figure.tight_layout()
    plot_path = out_dir / "plots" / f"{stamp}_tradeoff_curves.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    columns = ["mode", "tstart", "cfg_tar", "n", "lpaps", "lpaps_sem", "psnr", "ssim",
               "clap", "clap_sem", "muq", "muq_sem", "clap_dir", "muq_dir", "run"]
    frame[columns].to_csv(out_dir / f"{stamp}_hparam_sweep.csv", index=False)

    lines = [
        f"# AudioLDM2 editing hparam sweep -- {split} split",
        "",
        f"Generated {stamp} from `{root}` at commit "
        f"`{subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}`.",
        f"{len(frame)} scored configs, {int(frame['n'].iloc[0])} edits each.",
        "",
        f"Figure: `{plot_path}`",
        "",
        "## Pareto fronts (LPAPS down, metric up)",
        "",
    ]
    for y, title in Y_PANELS:
        lines += [f"### {title}", "", "| method | tstart | cfg_tar | LPAPS | " + title + " |",
                  "|---|---|---|---|---|"]
        for mode, group in frame.groupby("mode"):
            for _, row in pareto_front(group, y).iterrows():
                lines.append(
                    f"| {MODE_LABELS[mode]} | {row['tstart']} | {row['cfg_tar']:g} | "
                    f"{row['lpaps']:.3f} ± {row['lpaps_sem']:.3f} | "
                    f"{row[y]:.3f} ± {row[f'{y}_sem']:.3f} |"
                )
        lines.append("")

    lines += ["## Every config", "",
              "| method | tstart | cfg_src | cfg_tar | LPAPS | mel PSNR | mel SSIM | CLAP | MuQ | CLAP_dir | MuQ_dir |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(
            f"| {MODE_LABELS[row['mode']]} | {row['tstart']} | "
            f"{'-' if pd.isna(row['cfg_src']) else format(row['cfg_src'], 'g')} | {row['cfg_tar']:g} | "
            f"{row['lpaps']:.3f} ± {row['lpaps_sem']:.3f} | {row['psnr']:.2f} | {row['ssim']:.3f} | "
            f"{row['clap']:.3f} ± {row['clap_sem']:.3f} | {row['muq']:.3f} ± {row['muq_sem']:.3f} | "
            f"{row['clap_dir']:.3f} | {row['muq_dir']:.3f} |"
        )
    (out_dir / f"{stamp}_hparam_sweep.md").write_text("\n".join(lines) + "\n")

    with (out_dir / "run_meta.json").open("w") as handle:
        json.dump(
            {
                "timestamp": stamp,
                "runs_root": str(root),
                "pattern": pattern,
                "split": split,
                "num_configs": len(frame),
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {out_dir}")
    print(f"  {plot_path}")
    print(f"  {out_dir / f'{stamp}_hparam_sweep.md'}")


if __name__ == "__main__":
    fire.Fire(main)
