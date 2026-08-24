# ABOUTME: Generate Stable Audio Open DPMSolver latent trajectories from MusicCaps captions and
# ABOUTME: cache the frozen-teacher targets used to train the shifted-denoiser inversion LoRA.

import json
import sys
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.generate_trajectories import git_sha, load_captions  # noqa: E402
from src.inversion_lora.stable_audio import StableAudioTeacher, load_teacher  # noqa: E402


def save_sample(
    sample_dir: Path,
    trajectory: torch.Tensor,
    outputs: torch.Tensor,
    timesteps: list[float],
    text_audio: torch.Tensor,
    record: dict,
    meta: dict,
    store_dtype: torch.dtype,
) -> None:
    """Write one trajectory in the layout `AudioLDM2TrajectoryDataset` reads.

    Args:
        sample_dir: Destination directory for this sample.
        trajectory: Latents `[N + 1, C, L]`, noisiest first.
        outputs: Teacher predictions `[N, C, L]`, one per transition.
        timesteps: The sampling grid, one entry per transition.
        text_audio: Cross-attention states `[1, S, D]`.
        record: Caption record, written to `prompt.json`.
        meta: Run metadata; the per-sample shapes and counts are added here.
        store_dtype: Dtype the tensors are cast to on disk.
    """
    (sample_dir / "latents").mkdir(parents=True, exist_ok=True)
    (sample_dir / "targets").mkdir(parents=True, exist_ok=True)

    trajectory = trajectory.to(store_dtype)
    outputs = outputs.to(store_dtype)
    assert trajectory.shape[0] == outputs.shape[0] + 1, (trajectory.shape, outputs.shape)
    assert len(timesteps) == outputs.shape[0], (len(timesteps), outputs.shape)

    torch.save(trajectory, sample_dir / "latents/trajectory.pt")
    torch.save(outputs, sample_dir / "targets/target_eps.pt")
    torch.save({"text_audio": text_audio.to(store_dtype).cpu()[0]}, sample_dir / "conditioning.pt")
    with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
        json.dump(timesteps, f)
    with (sample_dir / "prompt.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    meta = {
        **meta,
        "trajectory_length": int(trajectory.shape[0]),
        "num_transitions": int(outputs.shape[0]),
        "latent_shape": list(trajectory.shape[1:]),
        "text_audio_shape": list(text_audio.shape[1:]),
    }
    # meta.json is the completion sentinel, so write it last and atomically: a crash between
    # tensor writes must not leave a directory that resume treats as finished.
    tmp_meta = sample_dir / "meta.json.tmp"
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(sample_dir / "meta.json")


def log_first_sample(
    teacher: StableAudioTeacher,
    record: dict,
    trajectory: torch.Tensor,
    outputs: torch.Tensor,
    timesteps: list[float],
    text_audio: torch.Tensor,
) -> None:
    """Print the shapes and magnitudes that show the first trajectory is wired correctly."""
    logger.info("First prompt: {!r}", record["prompt"])
    logger.info(
        "trajectory={} targets={} text_audio={} duration={:.2f}s timesteps[:3]={} [-3:]={}",
        tuple(trajectory.shape),
        tuple(outputs.shape),
        tuple(text_audio.shape),
        teacher.duration_s,
        timesteps[:3],
        timesteps[-3:],
    )
    logger.info(
        "latent RMS {:.4g} (noisiest) -> {:.4g} (cleanest); teacher RMS {:.4g} -> {:.4g}",
        float(trajectory[0].pow(2).mean().sqrt()),
        float(trajectory[-1].pow(2).mean().sqrt()),
        float(outputs[0].pow(2).mean().sqrt()),
        float(outputs[-1].pow(2).mean().sqrt()),
    )


@hydra.main(
    config_path="../../config",
    config_name="generate_trajectories_stable_audio",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Generate and cache Stable Audio Open trajectories for inversion-LoRA training."""
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type == "cpu":
        logger.warning("Running on CPU; use only for smoke tests.")
    else:
        torch.cuda.set_device(device)

    records = load_captions(Path(cfg.captions_csv), str(cfg.caption_column))
    start = int(cfg.start_index)
    end = len(records) if cfg.num_samples is None else start + int(cfg.num_samples)
    records = records[start:end]
    if not records:
        raise ValueError(f"No captions selected from range [{start}:{end})")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    logger.info("Loading {} ({} steps)", cfg.model_id, cfg.num_inference_steps)
    teacher = load_teacher(
        str(cfg.model_id),
        device,
        int(cfg.num_inference_steps),
        duration_s=cfg.duration_s,
        schedule=str(cfg.schedule),
    )
    scheduler = teacher.model.scheduler
    logger.info(
        "scheduler={} prediction_type={} timesteps={} ({:.4g}..{:.4g}) latent=[{}, {}]",
        type(scheduler).__name__,
        scheduler.config.prediction_type,
        len(scheduler.timesteps),
        float(scheduler.timesteps[0]),
        float(scheduler.timesteps[-1]),
        teacher.pipe.transformer.config.in_channels,
        teacher.latent_length,
    )

    store_dtype = getattr(torch, str(cfg.store_dtype))
    meta_base = {
        "model_id": str(cfg.model_id),
        "schedule": str(cfg.schedule),
        "solver": "first_order_ode",
        "pairing": "matched_timestep",
        "target_space": "data_prediction",
        "num_inference_steps": int(cfg.num_inference_steps),
        "duration_s": teacher.duration_s,
        "store_dtype": str(cfg.store_dtype),
        "git_sha": git_sha(),
    }

    for offset, record in enumerate(tqdm(records, desc="samples")):
        sample_idx = start + offset
        sample_dir = out_dir / f"sample_{sample_idx:06d}"
        if (sample_dir / "meta.json").exists() and not cfg.overwrite:
            continue

        seed = int(cfg.seed) + sample_idx
        text_audio = teacher.encode_prompt(record["prompt"])
        trajectory, data, grid = teacher.ode_trajectory(text_audio, seed=seed)

        # The pair inversion actually needs on this grid: the student sees the cleaner latent at
        # *its own* timestep and must predict the teacher's data prediction at the noisier one,
        # which is what the reverse step consumed. On an EDM grid the matched timestep beats the
        # shifted one 0.0179 to 0.0474 -- see output/sao_schedules/REPORT.md.
        # The last transition ends at sigma = 0, where the reverse step discards the sample and has
        # no inverse, so it is dropped: trajectory[:-1] pairs with data[:-1].
        trajectory, outputs, timesteps = trajectory[:-1], data[:-1], grid[1:]

        if offset == 0:
            log_first_sample(teacher, record, trajectory, outputs, timesteps, text_audio)

        save_sample(
            sample_dir,
            trajectory,
            outputs,
            timesteps,
            text_audio,
            record,
            {**meta_base, "sample_idx": sample_idx, "seed": seed},
            store_dtype,
        )

    logger.success("Wrote trajectories for {} samples to {}", len(records), out_dir)


if __name__ == "__main__":
    main()
