# ABOUTME: Build inversion-LoRA training pairs from REAL audio: forward-noise each clip's VAE
# ABOUTME: latent to every coarse grid sigma, query the teacher there, form exact-coarse-step inputs.

import json
import sys
from pathlib import Path

import hydra
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.generate_trajectories import git_sha  # noqa: E402
from src.inversion_lora.generate_trajectories_stable_audio import save_sample  # noqa: E402
from src.inversion_lora.stable_audio import (  # noqa: E402
    ExactDPMSolver,
    StableAudioTeacher,
    load_teacher,
)


def load_clip_records(csv_path: Path, audio_dir: Path, num_captions: int) -> list[dict]:
    """Caption rows whose MusicCaps clip is on disk, in caption-file order.

    Args:
        csv_path: `musiccaps-public.csv`.
        audio_dir: Directory of `[ytid]-[start-end].wav` clips.
        num_captions: How many leading caption rows to consider (the training slice).

    Returns:
        One record per available clip: prompt, ytid, and the wav path.
    """
    import pandas as pd

    meta = pd.read_csv(csv_path).head(num_captions)
    records = []
    for _, row in meta.iterrows():
        wav = audio_dir / f"[{row.ytid}]-[{int(row.start_s)}-{int(row.end_s)}].wav"
        if wav.exists():
            records.append({"prompt": str(row.caption), "ytid": str(row.ytid), "wav": str(wav)})
    return records


@torch.no_grad()
def encode_clip(teacher: StableAudioTeacher, wav_path: str) -> tuple[torch.Tensor, float]:
    """VAE-encode one clip, padded to the model's window like SAO's own training data.

    Args:
        teacher: The loaded teacher.
        wav_path: Path to the clip.

    Returns:
        `(latent [1, C, L], clip duration in seconds)`.
    """
    audio, sr = torchaudio.load(wav_path)
    target_sr = teacher.pipe.vae.config.sampling_rate
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] == 1:
        audio = audio.expand(2, -1)
    duration = min(audio.shape[1] / target_sr, teacher.max_duration_s)
    window = int(round(teacher.max_duration_s * target_sr))
    padded = torch.zeros(2, window)
    padded[:, : min(audio.shape[1], window)] = audio[:, :window]
    latent = teacher.pipe.vae.encode(padded[None].to(teacher.device)).latent_dist.mode()
    assert latent.shape[1:] == (
        teacher.pipe.transformer.config.in_channels,
        teacher.latent_length,
    ), latent.shape
    return latent, duration


@torch.no_grad()
def real_pairs(
    teacher: StableAudioTeacher,
    solver: ExactDPMSolver,
    x0: torch.Tensor,
    text_audio: torch.Tensor,
    seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Forward-noise a real latent to every coarse sigma and form exact-coarse-step pairs.

    States are `y_k = x0 + sigma_k * eps_k` with an independent draw per level (the forward
    marginal, the distribution the teacher itself was trained on). Each training input is one
    exact coarse reverse step from `y_k`, so the pair keeps the step invariant the coarse
    inversion relies on; the last coarse transition ends at sigma = 0 and is dropped.

    Args:
        teacher: The loaded teacher.
        solver: Solver over the coarse training grid.
        x0: Clean VAE latent `[1, C, L]`.
        text_audio: Cross-attention states for the clip's caption.
        seed: Seed for the noise draws of this sample.
        batch_size: DiT forward chunk size.

    Returns:
        `(rows, targets, states, timesteps)` in the on-disk layout `save_sample` expects:
        `rows[0]` is `y_0` and `rows[k + 1]` the input of pair k.
    """
    n = len(solver.timesteps)
    sigmas = solver.sigmas[:n].to(x0.device).reshape(n, 1, 1)
    generator = torch.Generator(device=x0.device).manual_seed(seed)
    eps = torch.randn((n, *x0.shape[1:]), generator=generator, device=x0.device)
    states = x0 + sigmas * eps

    timesteps = torch.tensor(solver.timesteps, device=x0.device)
    data = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        chunk_sigmas = sigmas[lo:hi]
        model_input = solver.model_input_batch(states[lo:hi], chunk_sigmas)
        raw = teacher.forward(model_input, timesteps[lo:hi], text_audio)
        data.append(solver.data_prediction_batch(states[lo:hi], raw, chunk_sigmas))
    data = torch.cat(data)
    assert data.shape == states.shape, (data.shape, states.shape)

    a, b = solver.coefficients_batch(torch.arange(n - 1))
    inputs = a.to(x0.device) * states[:-1] + b.to(x0.device) * data[:-1]
    rows = torch.cat([states[:1], inputs]).cpu()
    return rows, data[:-1].cpu(), states.cpu(), list(solver.timesteps)[1:]


@hydra.main(
    config_path="../../config", config_name="generate_real_pairs_stable_audio", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Generate the real-audio forward-noised training dataset."""
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type == "cpu":
        logger.warning("Running on CPU; use only for smoke tests.")
    else:
        torch.cuda.set_device(device)

    records = load_clip_records(
        Path(cfg.captions_csv), Path(cfg.audio_dir), int(cfg.num_captions)
    )
    if not records:
        raise FileNotFoundError(f"no clips from {cfg.captions_csv} found under {cfg.audio_dir}")
    draws = int(cfg.draws_per_clip)
    total = len(records) * draws
    start = int(cfg.start_index)
    end = total if cfg.num_samples is None else start + int(cfg.num_samples)
    logger.info("{} clips x {} draws = {} samples; this shard: [{}, {})",
                len(records), draws, total, start, min(end, total))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    logger.info("Loading {} ({} steps)", cfg.model_id, cfg.num_inference_steps)
    teacher = load_teacher(
        str(cfg.model_id), device, int(cfg.num_inference_steps), schedule=str(cfg.schedule)
    )
    solver = ExactDPMSolver(teacher.model.scheduler)

    meta_base = {
        "guidance_scale": 1.0,
        "model_id": str(cfg.model_id),
        "schedule": str(cfg.schedule),
        "solver": "first_order_ode",
        "pairing": "matched_timestep",
        "target_space": "data_prediction",
        "num_inference_steps": int(cfg.num_inference_steps),
        "input_space": "exact_coarse_step",
        "states_source": "real_audio_forward_noise",
        "duration_s": teacher.duration_s,
        "store_dtype": str(cfg.store_dtype),
        "git_sha": git_sha(),
    }
    store_dtype = getattr(torch, str(cfg.store_dtype))

    encoded: dict[int, tuple] = {}
    first_logged = False
    for sample_idx in tqdm(range(start, min(end, total)), desc="samples"):
        sample_dir = out_dir / f"sample_{sample_idx:06d}"
        if (sample_dir / "meta.json").exists() and not cfg.overwrite:
            continue
        clip_idx, draw = divmod(sample_idx, draws)
        record = records[clip_idx]
        if clip_idx not in encoded:
            teacher.set_duration(teacher.max_duration_s)
            x0, clip_duration = encode_clip(teacher, record["wav"])
            teacher.set_duration(clip_duration)
            encoded = {clip_idx: (x0, teacher.encode_prompt(record["prompt"]), clip_duration)}
        x0, text_audio, clip_duration = encoded[clip_idx]

        seed = int(cfg.seed_base) + sample_idx
        rows, targets, states, timesteps = real_pairs(
            teacher, solver, x0, text_audio, seed, int(cfg.forward_batch_size)
        )
        if not first_logged:
            gap = (rows[1:] - states[1:]).flatten(1).norm(dim=1) / \
                states[1:].flatten(1).norm(dim=1)
            logger.info(
                "First clip {!r} ({}): latent RMS {:.3g}, noisiest state RMS {:.3g}, "
                "input-vs-state gap rel mean {:.3e}",
                record["prompt"][:80], record["ytid"], float(x0.pow(2).mean().sqrt()),
                float(states[0].pow(2).mean().sqrt()), float(gap.mean()),
            )
            first_logged = True

        save_sample(
            sample_dir,
            rows,
            targets,
            timesteps,
            text_audio,
            {"prompt": record["prompt"], "ytid": record["ytid"], "wav": record["wav"]},
            {**meta_base, "sample_idx": sample_idx, "seed": seed, "clip_idx": clip_idx,
             "draw": draw, "clip_duration_s": clip_duration},
            store_dtype,
            states=states,
        )

    logger.success("shard [{}, {}) complete under {}", start, min(end, total), out_dir)


if __name__ == "__main__":
    main()
