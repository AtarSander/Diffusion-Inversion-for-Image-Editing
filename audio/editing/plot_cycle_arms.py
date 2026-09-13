# ABOUTME: Plots validation inversion loss for the cycle-loss arms against the no-cycle baseline,
# ABOUTME: showing that k=1 is inert at any lambda while k>=2 actually moves the solution.

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
VAL = re.compile(r"step (\d+) \{'val/loss': ([0-9.e+-]+)")
RUNS = {
    "no cycle (baseline)": ("train-5776575_0.err", "k", "-"),
    "k=1, ratio 0.1": ("train-5880219_29.err", "#9ecae1", "--"),
    "k=1, ratio 0.5": ("train-5880219_30.err", "#6baed6", "--"),
    "k=1, ratio 1.0": ("train-5880219_31.err", "#3182bd", "--"),
    "k=1, ratio 2.0": ("train-5880219_32.err", "#08519c", "--"),
    "k=2, ratio 1.0": ("train-5880219_33.err", "#d62728", "-"),
    "k=3, ratio 1.0": ("train-5880219_34.err", "#ff7f0e", "-"),
}


def read(log_dir: Path, name: str) -> dict[int, float]:
    """Validation loss by step, parsed from a training log."""
    p = log_dir / name
    if not p.exists():
        return {}
    text = p.read_bytes().decode("utf-8", "ignore").replace("\r", "\n")
    return {int(s): float(v) for s, v in VAL.findall(text)}


def main(log_dir: str, max_step: int = 5000, out_root: str = "output/cycle_arms") -> None:
    """Draw the two-panel comparison and write the numbers alongside it.

    Args:
        log_dir: Directory holding the slurm .err logs.
        max_step: Ignore checkpoints past this step, so arms are compared over a shared range.
        out_root: Destination, relative to `audio/`.
    """
    hist = {n: {s: v for s, v in read(Path(log_dir), f).items() if s <= max_step}
            for n, (f, _, _) in RUNS.items()}
    hist = {n: h for n, h in hist.items() if h}
    base = hist["no cycle (baseline)"]
    assert base, "baseline log not found"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle(
        "Cycle loss on Stable Audio: k=1 is inert at any weight, k>=2 moves the solution",
        fontsize=13, fontweight="bold", y=1.00,
    )
    for name, h in hist.items():
        _, color, style = RUNS[name]
        steps = sorted(h)
        ax1.plot(steps, [h[s] for s in steps], style, color=color, marker="o", ms=5, lw=1.8,
                 label=name)
        shared = [s for s in steps if s in base]
        if shared and name != "no cycle (baseline)":
            ax2.plot(shared, [abs(h[s] - base[s]) / base[s] for s in shared], style, color=color,
                     marker="o", ms=5, lw=1.8, label=name)

    ax1.set(xlabel="training step", ylabel="val inversion loss",
            title="Validation inversion loss")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8.5)
    ax2.axhline(1e-5, color="grey", ls=":", lw=1.2)
    ax2.text(0.98, 1.3e-5, "1e-5 — indistinguishable from baseline", ha="right", fontsize=8.5,
             color="grey", transform=ax2.get_yaxis_transform())
    ax2.set(xlabel="training step", ylabel="|arm - baseline| / baseline", yscale="log",
            title="Relative deviation from the no-cycle baseline")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"cycle_arms.{ext}", dpi=150, bbox_inches="tight")

    steps = sorted(set().union(*[set(h) for h in hist.values()]))
    lines = ["# Cycle-loss arms vs the no-cycle baseline — Stable Audio\n",
             f"Validation inversion loss (L_inv only, so every arm is scored on the same "
             f"quantity). Commit `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=AUDIO_ROOT, text=True).strip()[:8]}`.\n",
             "Figure: `cycle_arms.png`.\n", "| step | " + " | ".join(hist) + " |",
             "|" + "---|" * (len(hist) + 1)]
    for s in steps:
        lines.append(f"| {s} | " + " | ".join(
            f"{hist[n][s]:.6e}" if s in hist[n] else "—" for n in hist) + " |")
    k1 = [n for n in hist if n.startswith("k=1")]
    lines += ["\n## Spread across the four k=1 arms (20x span in target_ratio)\n",
              "| step | min | max | relative spread |", "|---|---|---|---|"]
    for s in steps:
        vals = [hist[n][s] for n in k1 if s in hist[n]]
        if len(vals) == len(k1):
            lines.append(f"| {s} | {min(vals):.9e} | {max(vals):.9e} | "
                         f"{(max(vals) - min(vals)) / min(vals):.2e} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "val_loss.json").write_text(json.dumps(hist, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
