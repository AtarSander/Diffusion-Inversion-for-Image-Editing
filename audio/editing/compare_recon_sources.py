# ABOUTME: Four-arm real-audio reconstruction on two sources — MedleyDB (benchmark) vs MusicCaps
# ABOUTME: (realfn's training-audio distribution) — the training-audio-source control.

import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "DDPM Inv.": ("ddpm", "#2ca02c"),
    "ODE Inv.": ("nolora", "#7f7f7f"),
    "ODE Inv.\nw/ LoRA": ("saocos_r8_a4_lr5e-5_checkpoint_step_4000", "#4c72b0"),
    "ODE Inv.\nw/ Real-Audio-LoRA": ("saocos_realfn_r8_a4_lr5e-5_checkpoint_step_3000", "#c44e52"),
}
SOURCES = {"train audio": "recon_mcrecon_s100", "benchmark audio": "recon_tracks_s100"}
FS = 14


def main(runs_root: str, out_root: str = "output/recon_sources") -> None:
    """Draw grouped mel-PSNR bars, one group per method, two bars per source.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
    """
    root = Path(runs_root)
    vals = {}
    for src, tag in SOURCES.items():
        vals[src] = {}
        for arm, (d, _) in ARMS.items():
            ps = root / f"stableaudio_acc_{tag}_{d}" / "psnr_ssim_per_file.csv"
            frame = pd.read_csv(ps)
            vals[src][arm] = (frame.psnr.mean(), frame.psnr.sem())

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.38
    xs = range(len(ARMS))
    hatches = [None, "//"]
    for s, src in enumerate(SOURCES):
        off = (s - 0.5) * width
        means = [vals[src][a][0] for a in ARMS]
        sems = [vals[src][a][1] for a in ARMS]
        ax.bar([x + off for x in xs], means, width, yerr=sems, capsize=4,
               color=[ARMS[a][1] for a in ARMS], edgecolor="black", linewidth=0.8,
               alpha=0.75 if s else 1.0, hatch=hatches[s], label=src)
        for x, m in zip(xs, means):
            ax.annotate(f"{m:.1f}", (x + off, m), ha="center", va="bottom", fontsize=FS - 3,
                        xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(ARMS), fontsize=FS - 1)
    ax.set_ylabel("mel PSNR (dB), higher = better", fontsize=FS)
    fig.suptitle("LoRA trained on real audio fails to generalize on benchmark",
                 fontsize=FS + 3, fontweight="bold", y=0.99)
    ax.set_title("Real Audio Reconstruction (PSNR)", fontsize=FS, y=1.0)
    ax.tick_params(labelsize=FS - 1)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax.legend(fontsize=FS - 1, loc="lower left")
    fig.tight_layout()

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / "plots" / f"recon_sources.{ext}", dpi=150, bbox_inches="tight")
    table = pd.DataFrame({src: {a: round(vals[src][a][0], 3) for a in ARMS} for src in SOURCES})
    table.to_csv(out / "recon_sources.csv")
    (out / "REPORT.md").write_text(
        "# Real-audio reconstruction, two sources, four arms (mel PSNR dB)\n\n"
        + table.to_markdown() + "\n")
    print(table.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
