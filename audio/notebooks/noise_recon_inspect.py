# ABOUTME: Jupyter-style inspection of the latest noise-recon benchmark reports: the per-arm table
# ABOUTME: for each model, the worst reconstructions per arm, and one source/reconstruction set.

# %%
# Parameters
from pathlib import Path

import pandas as pd

AUDIO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
report_root = AUDIO_ROOT / "output/noise_recon"

# %%
# Latest report per model
latest = {}
for run in sorted(report_root.glob("*_*")):
    latest[run.name.split("_", 2)[2]] = run
for model, run in latest.items():
    print(f"== {model}: {run}")
    print((run / "REPORT.md").read_text())

# %%
# Worst clips per arm (by LPAPS)
model = next(iter(latest))
frame = pd.read_csv(latest[model] / "per_sample.csv")
print(frame.groupby("arm")[["lpaps", "psnr", "ssim", "mae"]].describe().T)
print(frame.sort_values("lpaps", ascending=False).groupby("arm").head(3)[["arm", "index", "lpaps", "psnr", "prompt"]])

# %%
# Listen: one clip across arms (the wavs live in the benchmark run_dir, under outputs/)
import json

from IPython.display import Audio, display

index = 0
run_dir = Path(json.loads((latest[model] / "run_meta.json").read_text())["run_dir"])
for arm in ("source", "ddim", "ddim_lora", "ddpm", "gaussian"):
    print(arm)
    display(Audio(str(run_dir / "wavs" / arm / f"{index:04d}.wav")))
