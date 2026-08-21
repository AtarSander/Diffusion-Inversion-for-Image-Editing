# ABOUTME: Parse an inversion-LoRA training slurm log into its reconstruction and val-loss curves,
# ABOUTME: then emit the paired-against-baseline figure, a JSON record and a markdown mirror.

import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

BASELINE_RE = re.compile(r"Plain DDIM inversion baseline \(logged as step 0\): (\{.*?\})", re.S)
RECON_RE = re.compile(r"step (\d+) reconstruction (\{.*?\})", re.S)
VAL_RE = re.compile(r"step (\d+) (\{'val/loss'.*?\})", re.S)

# Measured on the frozen teacher over the same validation split in the earlier sweep. The slurm
# log only carries wandb's rounded 5e-5, which is consistent with it.
DISABLED_LOSS = 4.83e-5


def git_sha() -> str:
    """Current commit of this repository, or 'unknown' outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=AUDIO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def parse_log(log_path: Path) -> dict:
    """Pull the baseline and the per-step reconstruction and validation metrics out of a log.

    The log interleaves tqdm carriage returns with loguru lines, so each record is matched by
    regex over the whole file rather than line by line.

    Args:
        log_path: Slurm stderr of a `train.py` run.

    Returns:
        `baseline` dict, and `steps` mapping step to its merged reconstruction and val metrics.
    """
    text = log_path.read_text(errors="replace")

    baseline_match = BASELINE_RE.search(text)
    assert baseline_match, f"no plain-DDIM baseline line in {log_path}"
    baseline = ast.literal_eval(baseline_match.group(1))

    steps: dict[int, dict] = {}
    for pattern in (RECON_RE, VAL_RE):
        for step, payload in pattern.findall(text):
            steps.setdefault(int(step), {}).update(ast.literal_eval(payload))

    assert steps, f"no per-step metrics in {log_path}"
    return {"baseline": baseline, "steps": dict(sorted(steps.items()))}


def plot_curves(parsed: dict, out_path: Path, run_name: str) -> None:
    """Reconstruction against its plain-DDIM baseline, and val loss against the no-adapter loss.

    Args:
        parsed: Output of `parse_log`.
        out_path: Destination PNG.
        run_name: Title annotation.
    """
    steps = list(parsed["steps"])
    base = parsed["baseline"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))

    for key, label, color in (
        ("eval/real/mel_psnr", "real audio (13-60 s, edit geometry)", "#1f77b4"),
        ("eval/generated/mel_psnr", "generated (10.24 s)", "#d62728"),
    ):
        values = [parsed["steps"][s][key] for s in steps]
        left.plot(
            steps, values, marker="o", markersize=8, linewidth=2, color=color, label=label,
            markeredgecolor="black", markeredgewidth=0.8,
        )
        left.axhline(base[key], linestyle="--", linewidth=1.6, color=color, alpha=0.7)
        left.annotate(
            f"plain DDIM {base[key]:.2f} dB",
            xy=(steps[-1], base[key]), xytext=(-6, 6), textcoords="offset points",
            ha="right", fontsize=14, color=color,
        )

    left.set_xlabel("training step", fontsize=15)
    left.set_ylabel("mel PSNR (dB)", fontsize=15)
    left.set_title("Reconstruction: worse than no adapter after step 1000", fontsize=15)
    # Headroom below the lowest point so the legend cannot sit on top of a marker.
    low, high = left.get_ylim()
    left.set_ylim(low - 0.55 * (high - low), high)
    left.legend(fontsize=14, loc="lower left")
    left.grid(True, linestyle="--", alpha=0.2)
    left.tick_params(labelsize=14)

    # Percent of the shift gap closed, rather than raw val loss: the reference is 0% by
    # definition, so the readable range is not dominated by the distance to the no-adapter loss.
    for key, label, color in (
        ("val/loss", "adapter", "#2ca02c"),
        ("val/loss_ema", "adapter (EMA)", "#9467bd"),
    ):
        closed = [100 * (1 - parsed["steps"][s][key] / DISABLED_LOSS) for s in steps]
        right.plot(
            steps, closed, marker="s", markersize=8, linewidth=2, color=color, label=label,
            markeredgecolor="black", markeredgewidth=0.8,
        )

    right.axhspan(85, 88, color="black", alpha=0.10)
    right.annotate(
        "85-88% plateau held by every\nearlier preset, rank and lr",
        xy=(steps[len(steps) // 2], 86.5), fontsize=14, ha="center", va="center",
    )
    low, high = right.get_ylim()
    right.set_ylim(low - 0.08 * (high - low), high + 0.03 * (high - low))
    right.set_xlabel("training step", fontsize=15)
    right.set_ylabel("shift gap closed (%)", fontsize=15)
    right.set_title("Objective: best fit yet, then overfits", fontsize=15)
    right.legend(fontsize=14, loc="upper right")
    right.grid(True, linestyle="--", alpha=0.2)
    right.tick_params(labelsize=14)

    fig.suptitle(run_name, fontsize=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown(parsed: dict, out_path: Path, run_name: str, figure: Path) -> None:
    """Write the machine-readable mirror of the figure: both tables plus the derived deltas."""
    base = parsed["baseline"]
    lines = [
        f"# {run_name} — reconstruction and objective curves",
        "",
        f"Figure: `{figure.relative_to(out_path.parent)}`",
        "",
        "## Reconstruction, paired against plain DDIM on identical fixtures",
        "",
        "| step | real mel PSNR | vs baseline | generated mel PSNR | vs baseline |",
        "|---|---|---|---|---|",
        f"| 0 (plain DDIM) | {base['eval/real/mel_psnr']:.2f} | — "
        f"| {base['eval/generated/mel_psnr']:.2f} | — |",
    ]
    for step, metrics in parsed["steps"].items():
        real, gen = metrics["eval/real/mel_psnr"], metrics["eval/generated/mel_psnr"]
        lines.append(
            f"| {step} | {real:.2f} | {real - base['eval/real/mel_psnr']:+.2f} "
            f"| {gen:.2f} | {gen - base['eval/generated/mel_psnr']:+.2f} |"
        )

    lines += [
        "",
        "## Objective",
        "",
        f"No-adapter reference: {DISABLED_LOSS:.3e}",
        "",
        "| step | val loss | gap closed | val loss (EMA) | gap closed (EMA) |",
        "|---|---|---|---|---|",
    ]
    for step, metrics in parsed["steps"].items():
        raw, ema = metrics["val/loss"], metrics["val/loss_ema"]
        lines.append(
            f"| {step} | {raw:.3e} | {100 * (1 - raw / DISABLED_LOSS):.1f}% "
            f"| {ema:.3e} | {100 * (1 - ema / DISABLED_LOSS):.1f}% |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main(
    log_path: str,
    run_name: str = "q4_fullte_r32_a16_lr5e-4",
    out_root: str = "output/time_emb_lora",
) -> None:
    """Turn a training log into a figure, a JSON record and a markdown mirror.

    Args:
        log_path: Slurm stderr of the run.
        run_name: Name used in titles and filenames.
        out_root: Parent directory; a timestamped subdirectory is created under it.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = AUDIO_ROOT / out_root / stamp
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    parsed = parse_log(Path(log_path))
    print(f"parsed {len(parsed['steps'])} eval steps: {list(parsed['steps'])}")

    figure = out_dir / "plots" / f"{stamp}_{run_name}_curves.png"
    plot_curves(parsed, figure, run_name)
    write_markdown(parsed, out_dir / f"{stamp}_{run_name}_curves.md", run_name, figure)

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "git_sha": git_sha(),
                "run_name": run_name,
                "source_log": str(log_path),
                "command": " ".join(sys.argv),
                "disabled_loss_reference": DISABLED_LOSS,
                "metrics": parsed,
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
