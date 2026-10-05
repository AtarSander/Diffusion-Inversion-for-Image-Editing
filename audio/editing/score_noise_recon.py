# ABOUTME: Scores a noise_recon_benchmark run: noise normality (KL, top-k patch correlation) in
# ABOUTME: latent space, reconstruction (mel MAE/PSNR/SSIM, LPAPS) on decoded audio, plus a report.

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torchaudio
from audioldm_eval.datasets.load_mel import MelPairedDataset
from torch.utils.data import DataLoader

AUDIO_ROOT = Path(__file__).resolve().parents[1]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from editing.eval_medley import get_lpaps  # noqa: E402
from src.inversion_lora.noise_metrics import kl_div_per_dim, top_k_corr_in_patches  # noqa: E402
from src.metrics.alignment import mel_distances, mel_stft  # noqa: E402

# Row order and labels of the paper's inversion-quality table.
ARMS = {
    "gaussian": "Gaussian Noise",
    "ddim": "DDIM Inv.",
    "ddim_lora": "DDIM Inv. + LoRA",
    "ddpm": "DDPM Inv.",
}
MEL_SR = 32000  # the editing benchmark's paired-metric rate
PATCH_SIZE = 8
TOP_K = 20


def load_shards(run_dir: Path) -> dict:
    """Concatenate every shard of a run, checking they tile the sample range exactly once.

    Args:
        run_dir: The benchmark's `run_dir`.

    Returns:
        `indices`, `prompts`, `reference` and per-arm `noise`, sample-ordered.
    """
    metas = sorted(run_dir.glob("run_meta_shard*.json"))
    paths = sorted((run_dir / "latents").glob("shard*.pt"))
    assert metas and len(paths) == len(metas), (
        f"{len(paths)} shards but {len(metas)} run_meta files"
    )
    config = json.loads(metas[0].read_text())["config"]
    expected = config["num_shards"] if config["limit"] is None else len(paths)
    assert len(paths) == expected, f"{len(paths)} of {expected} shards present in {run_dir}"

    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    order = np.argsort(np.concatenate([s["indices"] for s in shards]))
    indices = np.concatenate([s["indices"] for s in shards])[order]
    assert len(set(indices.tolist())) == len(indices), "a sample appears in two shards"
    if config["limit"] is None:
        assert indices.tolist() == list(range(config["num_prompts"] * config["seeds_per_prompt"]))
    return {
        "config": config,
        "git_sha": json.loads(metas[0].read_text())["git_sha"],
        "indices": indices.tolist(),
        "prompts": [[p for s in shards for p in s["prompts"]][i] for i in order],
        "reference": torch.cat([s["reference"] for s in shards])[order],
        "noise": {arm: torch.cat([s["noise"][arm] for s in shards])[order] for arm in ARMS},
    }


def as_patch_layout(latents: torch.Tensor) -> torch.Tensor:
    """View latents as `[N, C, H, W]` for the patch metrics.

    A 1-D Stable Audio latent `[N, C, L]` becomes `[N, C, L, 1]`, so an 8x8 patch is 8 frames x
    every channel -- the same all-channel patch an AudioLDM2 latent gives over 8x8 of (time, freq).
    """
    return latents if latents.ndim == 4 else latents.unsqueeze(-1)


def normality(reference: torch.Tensor, latents: torch.Tensor) -> dict[str, float]:
    """KL to the generation noise and the top-k patch correlation of one arm's noise."""
    reference, latents = as_patch_layout(reference), as_patch_layout(latents)
    assert reference.shape == latents.shape, (reference.shape, latents.shape)
    corr = top_k_corr_in_patches(latents, patch_size=PATCH_SIZE, top_k=TOP_K)
    return {
        "kl": kl_div_per_dim(reference, latents),
        "corr": corr["mean"],
        "corr_std": corr["std"],
        "noise_std": float(latents.std()),
        "noise_mean": float(latents.mean()),
    }


def mel_scores(arm_dir: Path, source_dir: Path) -> pd.DataFrame:
    """Per-file mel PSNR/SSIM/MAE of an arm against the source clips, as the benchmark scores them.

    `MelPairedDataset` takes the first channel, removes DC and resamples to 32 kHz itself, which
    is what the editing eval does after its on-disk resample.
    """
    loader = DataLoader(
        MelPairedDataset(str(arm_dir), str(source_dir), mel_stft(MEL_SR), MEL_SR),
        batch_size=1,
        num_workers=8,
    )
    rows = {}
    for mel_gen, mel_target, filename, _ in loader:
        scores = mel_distances(mel_gen.numpy()[0], mel_target.numpy()[0])
        assert np.isfinite(scores["psnr"]), f"{filename[0]}: bit-identical mels, PSNR is infinite"
        rows[int(Path(filename[0]).stem)] = scores
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def lpaps_scores(arm_dir: Path, source_dir: Path, indices: list[int], device) -> np.ndarray:
    """Per-file LPAPS of an arm against the source clips, through the editing eval's `get_lpaps`."""
    sources, arms, rates = [], [], []
    for index in indices:
        source, sr_source = torchaudio.load(str(source_dir / f"{index:04d}.wav"))
        audio, sr_audio = torchaudio.load(str(arm_dir / f"{index:04d}.wav"))
        assert sr_source == sr_audio, (sr_source, sr_audio)
        sources.append(source)
        arms.append(audio)
        rates.append(sr_source)
    frame = get_lpaps(sources, arms, rates, rates, device)
    return frame.sort_values("audio_idx")["lpaps"].to_numpy()


def git_sha() -> str:
    """Current commit, for the report's provenance."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AUDIO_ROOT, text=True).strip()


def report(label: str, run_dir: Path, data: dict, table: pd.DataFrame) -> str:
    """The markdown mirror of the results: the paper-style table plus provenance."""
    cfg = data["config"]
    lines = [
        f"# Noise normality vs reconstruction — {label}",
        "",
        f"- run: `{run_dir}` (generated at git `{data['git_sha'][:10]}`, scored at `{git_sha()[:10]}`)",
        f"- N = {len(data['indices'])} clips ({cfg['num_prompts']} MedleyMD captions x "
        f"{cfg['seeds_per_prompt']} seeds), {cfg['num_inference_steps']} inversion and denoising "
        f"steps, CFG {cfg['guidance_scale']}, {cfg['duration_s']} s, model `{cfg['model_id']}`",
        f"- LoRA (inversion pass only): `{cfg['lora_path']}`",
        "- Normality in latent space against the generation noise: KL = per-dimension Gaussian KL "
        "averaged over dimensions (x100); Corr = mean top-20 |Pearson| within 8x8 latent patches "
        "(Stable Audio: 8 frames x all channels). Gaussian Noise is a fresh draw, i.e. the null.",
        "- Reconstruction on decoded audio against the generated source clip: mel MAE/PSNR/SSIM "
        "(audioldm_eval mel at 32 kHz, as in the editing benchmark) and LPAPS (CLAP, 10 s windows). "
        "Gaussian Noise reconstructs nothing: it is the unrelated-sample ceiling.",
        "",
        "| Method | Corr ↓ | KL ×10² ↓ | MAE ↓ | LPAPS ↓ | PSNR ↑ | SSIM ↑ | noise std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['method']} | {row['corr']:.4f} | {100 * row['kl']:.4f} "
            f"| {row['mae']:.4f} ± {row['mae_sem']:.4f} | {row['lpaps']:.3f} ± {row['lpaps_sem']:.3f} "
            f"| {row['psnr']:.2f} ± {row['psnr_sem']:.2f} | {row['ssim']:.4f} ± {row['ssim_sem']:.4f} "
            f"| {row['noise_std']:.4f} |"
        )
    lines += [
        "",
        f"Reference (generation noise) Corr = {data['reference_corr']:.4f}: the finite-N floor "
        "every Corr above should be read against.",
        "",
        "± is the standard error over clips. Per-clip values: `per_sample.csv`.",
    ]
    return "\n".join(lines) + "\n"


def main(run_dir: str, label: str, out_root: str = "output/noise_recon") -> None:
    """Score every arm of a benchmark run and write metrics.json, per_sample.csv and REPORT.md.

    Args:
        run_dir: The benchmark's `run_dir`, relative to audio/ or absolute.
        label: Model label for the report and the output directory name.
        out_root: Where the timestamped output directory is created.
    """
    run_path = (AUDIO_ROOT / run_dir).resolve()
    data = load_shards(run_path)
    indices = data["indices"]
    print(f"{len(indices)} samples, latents {tuple(data['reference'].shape)}")
    data["reference_corr"] = top_k_corr_in_patches(
        as_patch_layout(data["reference"]), patch_size=PATCH_SIZE, top_k=TOP_K
    )["mean"]
    print(f"reference noise corr {data['reference_corr']:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_dir = run_path / "wavs/source"
    rows, per_sample = [], []
    for arm, method in ARMS.items():
        arm_dir = run_path / "wavs" / arm
        stats = normality(data["reference"], data["noise"][arm])
        mel = mel_scores(arm_dir, source_dir)
        assert mel.index.tolist() == indices, f"{arm}: mel pairs do not cover the samples"
        mel["lpaps"] = lpaps_scores(arm_dir, source_dir, indices, device)
        rows.append(
            {
                "arm": arm,
                "method": method,
                **stats,
                **{m: float(mel[m].mean()) for m in ("mae", "lpaps", "psnr", "ssim")},
                **{f"{m}_sem": float(mel[m].sem()) for m in ("mae", "lpaps", "psnr", "ssim")},
            }
        )
        print(
            f"{method}: corr {stats['corr']:.4f} kl {stats['kl']:.5f} mae {rows[-1]['mae']:.4f} "
            f"lpaps {rows[-1]['lpaps']:.3f} psnr {rows[-1]['psnr']:.2f} ssim {rows[-1]['ssim']:.4f}"
        )
        per_sample.append(mel.assign(arm=arm, prompt=data["prompts"]).rename_axis("index"))

    table = pd.DataFrame(rows)
    out_dir = AUDIO_ROOT / out_root / f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
    out_dir.mkdir(parents=True)
    pd.concat(per_sample).to_csv(out_dir / "per_sample.csv")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"reference_corr": data["reference_corr"], "arms": table.to_dict("records")}, indent=2
        )
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_path),
                "generation_config": data["config"],
                "generation_git_sha": data["git_sha"],
                "scoring_git_sha": git_sha(),
                "command": " ".join(sys.argv),
                "scored_at": datetime.now().isoformat(),
                "n": len(indices),
            },
            indent=2,
        )
    )
    (out_dir / "REPORT.md").write_text(report(label, run_path, data, table))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
