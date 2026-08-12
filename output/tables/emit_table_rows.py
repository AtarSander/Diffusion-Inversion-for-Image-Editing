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
]


def stats(run: Path) -> dict[str, tuple[float, float]]:
    """Mean and standard error for every per-example metric of one run."""
    out = {}
    for label, filename, column in COLUMNS:
        series = pd.read_csv(run / filename, index_col=0)[column]
        assert len(series) == 696, f"{run.name}/{filename}: {len(series)} rows, expected 696"
        out[label] = (series.mean(), series.sem())
    distances = json.loads((run / "source_distance_metrics.json").read_text())
    out["psnr"] = (float(distances["psnr"]), None)
    out["ssim"] = (float(distances["ssim"]), None)
    return out


results = {
    model: {method: stats(MOUNT / rel) for method, rel in runs}
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
    print(f"    \\multirow{{3}}{{*}}{{{model}}}")
    for method in methods:
        cells = []
        for metric in ORDER:
            mean, sem = methods[method][metric]
            digits = 2 if metric == "psnr" else 3
            body = f"{mean:.{digits}f}" if sem is None else f"{mean:.3f}_{{\\pm {sem:.3f}}}"
            if methods[method][metric][0] == methods[best[metric]][metric][0]:
                body = (
                    f"\\mathbf{{{mean:.{digits}f}}}"
                    if sem is None
                    else f"\\mathbf{{{mean:.3f}}}_{{\\pm {sem:.3f}}}"
                )
            cells.append(f"${body}$")
        print(f"      & {method:8s} & " + " & ".join(cells) + r" \\")
    print(r"    \midrule")
