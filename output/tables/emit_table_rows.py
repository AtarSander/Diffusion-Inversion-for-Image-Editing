# ABOUTME: Emit the LaTeX body rows for the MedleyMD results table with SEM instead of SD,
# ABOUTME: computed from the per-example CSVs so n is counted rather than assumed.

import json
from pathlib import Path

import pandas as pd

MOUNT = Path("/nas/lstanisz/code/lorainv/pwr-mount/audio/outputs/edits/medleymd")

MODELS = {
    "AudioLDM2-large": [
        ("DDPM-inv", "audioldm2_ddpm/audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200"),
        ("DDIM-inv", "audioldm2_ddim/audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200"),
        ("SDEdit", "audioldm2_sdedit/audioldm2_sdedit_cfgtar12.0_t100_s200"),
    ],
    "AudioLDM2-large, DDIM inversion variants": [
        ("DDIM, $w_{src}$=3.0", "audioldm2_ddim/audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200"),
        ("DDIM, $w_{src}$=1.0", "audioldm2_ddim/audioldm2_ddim_cfgsrc1.0_cfgtar12.0_t200_s200"),
        ("+ LoRA attn r8 (6k)",
         "audioldm2_ddim/audioldm2_ddimlora_attn_r8_a4_lr2e-4_checkpoint_step_6000"),
        ("+ LoRA attn r32 (20k)",
         "audioldm2_ddim/audioldm2_ddimlora_r32_a16_lr2e-4_checkpoint_final"),
        ("+ LoRA full r32 (6k)",
         "audioldm2_ddim/audioldm2_ddimlora_full_r32_a16_lr5e-4_checkpoint_step_6000"),
    ],
    "Stable Audio Open": [
        ("DDPM-inv", "medleymd/stable_audio/stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100"),
        ("DDIM-inv", "medleymd/stable_audio/stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100"),
        ("SDEdit", "medleymd/stable_audio/stableaudio_sdedit_cfgtar3.5_t50_s100"),
    ],
}

# metric label -> (csv file, column)
COLUMNS = [
    ("LPAPS", "lpaps_to_source.csv", "lpaps"),
    ("CLAP", "clap_to_target_prompt.csv", "clap"),
    ("MuLan", "mulan_to_target_prompt.csv", "muqt_sim_p0"),
    ("CLAP_dir", "directional_to_prompts.csv", "clap_dir"),
    ("MuLan_dir", "directional_to_prompts.csv", "mulan_dir"),
    ("psnr", "psnr_ssim_per_file.csv", "psnr"),
    ("ssim", "psnr_ssim_per_file.csv", "ssim"),
]


def stats(run: Path) -> dict[str, tuple[float, float]]:
    """Mean and standard error for every per-example metric of one run."""
    out = {}
    for label, filename, column in COLUMNS:
        series = pd.read_csv(run / filename, index_col=0)[column]
        assert len(series) == 696, f"{run.name}/{filename}: {len(series)} rows, expected 696"
        out[label] = (series.mean(), series.sem())

    # The aggregates the eval reported must match what the per-file CSVs say, or the table and
    # source_distance_metrics.json would disagree about the same run.
    distances = json.loads((run / "source_distance_metrics.json").read_text())
    for key in ("psnr", "ssim"):
        assert abs(out[key][0] - float(distances[key])) < 5e-4, (
            f"{run.name}: {key} per-file mean {out[key][0]:.4f} != reported {distances[key]}"
        )
        assert abs(out[key][1] - float(distances[f"{key}_sem"])) < 5e-4, (
            f"{run.name}: {key} per-file sem {out[key][1]:.4f} != reported {distances[f'{key}_sem']}"
        )
    return out


def safe_stats(rel: str):
    """Stats for a run, or None when it has not been scored yet."""
    try:
        return stats(MOUNT / rel)
    except (FileNotFoundError, AssertionError) as exc:
        print(f"% skipping {rel}: {exc}")
        return None


results = {
    model: {
        method: values
        for method, rel in runs
        if (values := safe_stats(rel)) is not None
    }
    for model, runs in MODELS.items()
}

# Bold the best value per column within each model panel.
LOWER_IS_BETTER = {"LPAPS"}
ORDER = ["LPAPS", "psnr", "ssim", "CLAP", "MuLan", "CLAP_dir", "MuLan_dir"]

for model, methods in results.items():
    best = {}
    for metric in ORDER:
        pick = min if metric in LOWER_IS_BETTER else max
        best[metric] = pick(methods, key=lambda m: methods[m][metric][0])
    print(f"    \\multirow{{{len(methods)}}}{{*}}{{{model}}}")
    for method in methods:
        cells = []
        for metric in ORDER:
            mean, sem = methods[method][metric]
            value = f"\\mathbf{{{mean:.3f}}}" if method == best[metric] else f"{mean:.3f}"
            cells.append(f"${value}_{{\\pm {sem:.3f}}}$")
        print(f"      & {method:8s} & " + " & ".join(cells) + r" \\")
    print(r"    \midrule")
