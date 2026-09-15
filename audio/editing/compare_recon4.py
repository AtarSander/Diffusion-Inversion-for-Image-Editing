# ABOUTME: The four-arm real-audio reconstruction figure: DDPM reference, no-LoRA ODEInv, the
# ABOUTME: trajectory adapter and the realfn adapter — the chain-consistency gate result.

import json
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
    "DDPM-inv": ("ddpm", "#2ca02c"),
    "ODEInv, no LoRA": ("nolora", "#7f7f7f"),
    "ODEInv + trajectory LoRA": ("saocos_r8_a4_lr5e-5_checkpoint_step_4000", "#4c72b0"),
    "ODEInv + real-fwd-noise LoRA": ("saocos_realfn_r8_a4_lr5e-5_checkpoint_step_3000",
                                      "#c44e52"),
}
PANELS = [("psnr", "mel PSNR (dB), higher = better", "psnr_ssim"),
          ("lpaps", "LPAPS to source, lower = better", "per_example"),
          ("clap", "CLAP to source caption, higher = better", "per_example")]
FS = 14


def main(runs_root: str, out_root: str = "output/recon4") -> None:
    """Draw the reconstruction comparison.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
    """
    root = Path(runs_root)
    data = {}
    for arm, (d, _) in ARMS.items():
        p = root / f"stableaudio_acc_recon_tracks_s100_{d}"
        data[arm] = {"per_example": pd.read_csv(p / "per_example_metrics.csv"),
                     "psnr_ssim": pd.read_csv(p / "psnr_ssim_per_file.csv")}

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.0 * len(PANELS), 5.6))
    fig.suptitle("Real-audio reconstruction (invert 99 steps, denoise with the source caption, "
                 "35 MedleyDB tracks)", fontsize=FS + 3, fontweight="bold")
    for ax, (col, label, table) in zip(axes, PANELS):
        for i, (arm, (_, color)) in enumerate(ARMS.items()):
            vals = data[arm][table][col]
            ax.bar(i, vals.mean(), 0.7, yerr=vals.sem(), capsize=4, color=color,
                   edgecolor="black", linewidth=0.8)
            ax.annotate(f"{vals.mean():.2f}" if col == "psnr" else f"{vals.mean():.3f}",
                        (i, vals.mean()), ha="center", va="bottom", fontsize=FS,
                        xytext=(0, 6), textcoords="offset points")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(list(ARMS), fontsize=FS - 1, rotation=18, ha="right")
        ax.set_ylabel(label, fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / "plots" / f"recon4.{ext}", dpi=150, bbox_inches="tight")

    summary = pd.DataFrame(
        {arm: {"mel_psnr": data[arm]["psnr_ssim"]["psnr"].mean(),
               "mel_ssim": data[arm]["psnr_ssim"]["ssim"].mean(),
               "lpaps": data[arm]["per_example"]["lpaps"].mean(),
               "clap_src": data[arm]["per_example"]["clap"].mean()}
         for arm in ARMS}).T
    summary.to_csv(out / "recon4.csv")
    (out / "REPORT.md").write_text(
        "# Four-arm real-audio reconstruction (chain-consistency gate)\n\n"
        "Figure: `plots/recon4.png`. 35 MedleyDB tracks, t99, cfg 1.0, source caption.\n\n"
        + summary.to_markdown(floatfmt=".4f") + "\n")
    print(json.dumps({k: round(v, 3) for k, v in summary["mel_psnr"].items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
