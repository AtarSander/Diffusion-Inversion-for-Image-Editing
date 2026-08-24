# ABOUTME: Paired comparison of the Stable Audio inversion LoRA against its no-LoRA twins, on both
# ABOUTME: the reconstruction ladder and the editing grid, writing a markdown report and its JSON.

import json
from datetime import datetime
from pathlib import Path

import fire
import pandas as pd
from scipy import stats

CHECKPOINT_DIR = "sao_r8_a4_lr5e-5"
LADDER = ["nolora", 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, "18000_ema"]
EDIT_ARMS = ["18000", "18000_ema"]
TSTART = [25, 50, 75, 100]
CFG_TAR = [3.5, 7.0]


def read_run(run_dir: Path) -> dict:
    """Load one scored run's aggregate metrics and its two per-example frames.

    Args:
        run_dir: Directory holding `metrics.json`, `per_example_metrics.csv` and
            `psnr_ssim_per_file.csv`, as written by the eval job.

    Returns:
        Dict with `agg`, `per` (LPAPS/CLAP per row) and `psnr` (mel PSNR/SSIM per file).
    """
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"{run_dir} has no metrics.json; has the eval job scored it?")
    return {
        "agg": json.loads((run_dir / "metrics.json").read_text()),
        "per": pd.read_csv(run_dir / "per_example_metrics.csv"),
        "psnr": pd.read_csv(run_dir / "psnr_ssim_per_file.csv", index_col=0),
    }


def paired(arm: dict, base: dict, column: str, frame: str) -> tuple[float, float]:
    """Mean paired difference and its two-sided t-test p-value, over the shared rows.

    Args:
        arm: Run being tested, from `read_run`.
        base: Reference run.
        column: Metric column.
        frame: Which frame to read, `per` or `psnr`.

    Returns:
        `(mean difference, p value)`, arm minus base.
    """
    a, b = arm[frame], base[frame]
    if frame == "psnr":
        shared = a.index.intersection(b.index)
        a, b = a.loc[shared, column], b.loc[shared, column]
    else:
        a, b = a[column], b[column]
    assert len(a) == len(b) and len(a) > 1, f"{column}: {len(a)} vs {len(b)} paired rows"
    return float((a - b).mean()), float(stats.ttest_rel(a, b).pvalue)


def ladder_table(root: Path) -> tuple[str, dict]:
    """Reconstruction dose-response: every checkpoint against the frozen teacher."""
    base = read_run(root / "stableaudio_recon_tracks_s100_nolora")
    rows, records = [], {}
    header = (
        "| arm | LPAPS | d | p | mel PSNR | d | p | SSIM | d |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for arm in LADDER:
        name = "nolora" if arm == "nolora" else f"checkpoint_step_{arm}"
        run = read_run(root / f"stableaudio_recon_tracks_s100_{name}")
        lpaps = run["agg"]["final"]["LPAPS"]["mean"]
        psnr, ssim = (float(run["agg"]["source_distance"][k]) for k in ("psnr", "ssim"))
        if arm == "nolora":
            rows.append(f"| {arm} | {lpaps:.3f} | -- | -- | {psnr:.3f} | -- | -- | {ssim:.3f} | -- |")
            records[str(arm)] = {"lpaps": lpaps, "psnr": psnr, "ssim": ssim}
            continue
        dl, pl = paired(run, base, "lpaps", "per")
        dp, pp = paired(run, base, "psnr", "psnr")
        ds, _ = paired(run, base, "ssim", "psnr")
        rows.append(
            f"| {arm} | {lpaps:.3f} | {dl:+.3f} | {pl:.3f} | {psnr:.3f} | {dp:+.3f} | {pp:.3f} "
            f"| {ssim:.3f} | {ds:+.4f} |"
        )
        records[str(arm)] = {
            "lpaps": lpaps, "d_lpaps": dl, "p_lpaps": pl,
            "psnr": psnr, "d_psnr": dp, "p_psnr": pp, "ssim": ssim, "d_ssim": ds,
        }
    return header + "\n" + "\n".join(rows), records


def grid_table(root: Path) -> tuple[str, dict]:
    """Editing grid: each adapter against its no-LoRA twin at identical settings."""
    rows, records = [], {}
    header = (
        "| cell | no-LoRA LPAPS | arm | d LPAPS | p | d CLAP | p | d mel PSNR | p |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for tstart in TSTART:
        for cfg in CFG_TAR:
            cell = f"t{tstart}/cfg{cfg}"
            base = read_run(root / f"stableaudio_ddim_nolora_hparam_cfgtar{cfg}_t{tstart}_s100")
            records[cell] = {"nolora": base["agg"]["final"]["LPAPS"]["mean"]}
            for arm in EDIT_ARMS:
                run = read_run(
                    root / f"stableaudio_ddimlora_hparam_{CHECKPOINT_DIR}_checkpoint_step_{arm}"
                    f"_cfgtar{cfg}_t{tstart}_s100"
                )
                dl, pl = paired(run, base, "lpaps", "per")
                dc, pc = paired(run, base, "clap", "per")
                dp, pp = paired(run, base, "psnr", "psnr")
                rows.append(
                    f"| {cell} | {records[cell]['nolora']:.3f} | {arm} | {dl:+.4f} | {pl:.3f} "
                    f"| {dc:+.4f} | {pc:.3f} | {dp:+.3f} | {pp:.3f} |"
                )
                records[cell][arm] = {
                    "d_lpaps": dl, "p_lpaps": pl, "d_clap": dc,
                    "p_clap": pc, "d_psnr": dp, "p_psnr": pp,
                }
    return header + "\n" + "\n".join(rows), records


def main(root: str, output_dir: str = "output/sao_lora") -> None:
    """Write the paired reconstruction and editing tables for the Stable Audio adapter.

    Args:
        root: The `medleymd/stable_audio` edits directory holding the scored runs.
        output_dir: Destination for the timestamped report, relative to `audio/`.
    """
    root = Path(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parents[1] / output_dir / stamp
    out.mkdir(parents=True, exist_ok=True)

    ladder_md, ladder_json = ladder_table(root)
    grid_md, grid_json = grid_table(root)
    print(ladder_md, "\n")
    print(grid_md)

    report = f"""# Stable Audio inversion LoRA: reconstruction and editing ({stamp})

Adapter `{CHECKPOINT_DIR}` (attn preset, r8, lr 5e-5), applied to the DPMSolver inversion pass
only. Runs read from `{root}`.

## Reconstruction, 35 distinct MedleyDB tracks

Full inversion (tstart = steps = 100) at cfg_tar=1.0 with the source caption, so the output should
be the input. `d` is the paired difference against the frozen teacher over the same tracks.

{ladder_md}

## Editing, 115-row hparam split

Each adapter against its no-LoRA twin at identical settings, paired over the same rows.

{grid_md}
"""
    (out / "REPORT.md").write_text(report)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"root": str(root), "ladder": ladder_json, "grid": grid_json}, f, indent=2)
    print(f"\nwrote {out}/REPORT.md and metrics.json")


if __name__ == "__main__":
    fire.Fire(main)
