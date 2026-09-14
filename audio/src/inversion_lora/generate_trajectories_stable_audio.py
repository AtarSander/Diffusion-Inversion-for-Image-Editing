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
from src.inversion_lora.stable_audio import (  # noqa: E402
    ExactDPMSolver,
    StableAudioTeacher,
    load_teacher,
)


def dense_coarse_pairs(
    trajectory: torch.Tensor,
    data: torch.Tensor,
    coarse_solver: ExactDPMSolver,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce a fine ODE trajectory to coarse-grid training rows with exact-coarse-step inputs.

    The fine trajectory's states at the coarse grid points are higher quality than a coarse
    trajectory's, but `stride` fine steps are not one coarse step, so the fine state at point
    k+1 is NOT what the coarse inverse step will later have to undo. Each training input is
    therefore formed as exactly one coarse reverse step from the fine state at point k: the row
    then satisfies the same exact step invariant as a plain coarse dataset, while both sides of
    the pair sit on the dense trajectory. The last coarse transition ends at sigma = 0 and is
    dropped, as in the plain pipeline.

    Args:
        trajectory: Fine latents `[F + 1, C, L]`, noisiest first.
        data: Data predictions `[F, C, L]`, `data[i]` at `(trajectory[i], fine timestep i)`.
        coarse_solver: Solver over the coarse grid the LoRA trains and inverts on.
        stride: Fine steps per coarse step; fine point `stride * k` is coarse point `k`.

    Returns:
        `(rows, targets, states, gap_rel)`: `rows[0]` is the initial latent and `rows[k + 1]`
        the input of pair k `[K + 1, C, L]`; `targets[k]` the data prediction at `states[k]`
        `[K, C, L]`; `states` the fine latents at every coarse point `[K + 1, C, L]`, kept for
        verification; `gap_rel[k]` the relative distance between `rows[k + 1]` and the fine
        state at coarse point k+1 — the dense-vs-coarse quality delta this dataset banks on.
    """
    n_coarse = len(coarse_solver.timesteps)
    assert trajectory.shape[0] > stride * (n_coarse - 1), (trajectory.shape, stride, n_coarse)
    states = trajectory[:: stride][:n_coarse]
    targets = data[:: stride][: n_coarse - 1]
    assert states.shape[0] == n_coarse and targets.shape[0] == n_coarse - 1

    a, b = coarse_solver.coefficients_batch(torch.arange(n_coarse - 1))
    inputs = a * states[:-1] + b * targets
    rows = torch.cat([states[:1], inputs])

    gap = inputs - states[1:]
    gap_rel = gap.flatten(1).norm(dim=1) / states[1:].flatten(1).norm(dim=1)
    return rows, targets, states, gap_rel


def save_sample(
    sample_dir: Path,
    trajectory: torch.Tensor,
    outputs: torch.Tensor,
    timesteps: list[float],
    text_audio: torch.Tensor,
    record: dict,
    meta: dict,
    store_dtype: torch.dtype,
    uncond: torch.Tensor | None = None,
    states: torch.Tensor | None = None,
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
        uncond: Unconditional predictions `[N, C, L]` when the trajectory was sampled with
            guidance; the dataset reads these as `uncond_eps` and the trainer recombines them.
        states: Fine latents at the coarse grid points `[N + 1, C, L]` for a dense dataset;
            training never reads them, verification recomputes the step invariant from them.
    """
    (sample_dir / "latents").mkdir(parents=True, exist_ok=True)
    (sample_dir / "targets").mkdir(parents=True, exist_ok=True)

    trajectory = trajectory.to(store_dtype)
    outputs = outputs.to(store_dtype)
    assert trajectory.shape[0] == outputs.shape[0] + 1, (trajectory.shape, outputs.shape)
    assert len(timesteps) == outputs.shape[0], (len(timesteps), outputs.shape)

    torch.save(trajectory, sample_dir / "latents/trajectory.pt")
    torch.save(outputs, sample_dir / "targets/target_eps.pt")
    if states is not None:
        assert states.shape == trajectory.shape, (states.shape, trajectory.shape)
        torch.save(states.to(store_dtype), sample_dir / "latents/states.pt")
    if uncond is not None:
        assert uncond.shape == outputs.shape, (uncond.shape, outputs.shape)
        torch.save(uncond.to(store_dtype), sample_dir / "targets/uncond_eps.pt")
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
    guidance_scale = float(cfg.get("guidance_scale", 1.0))
    uncond_audio = teacher.encode_prompt("") if guidance_scale != 1.0 else None

    # train_stride > 1: sample on a dense grid, train on the usual coarse one. The fine grid must
    # nest the coarse grid exactly -- linspace ramps share points when F = stride * (K - 1) + 1,
    # e.g. 991 fine points for a 100-point coarse grid at stride 10.
    train_stride = int(cfg.get("train_stride", 1))
    coarse_solver = None
    if train_stride > 1:
        assert guidance_scale == 1.0, "dense sampling is wired for the unguided objective only"
        fine_steps = int(cfg.num_inference_steps)
        assert (fine_steps - 1) % train_stride == 0, (
            f"num_inference_steps={fine_steps} does not nest a coarse grid at stride "
            f"{train_stride}; use stride * (coarse - 1) + 1 points (991 for 100 @ 10)"
        )
        coarse_steps = (fine_steps - 1) // train_stride + 1
        coarse_scheduler = type(scheduler).from_config(scheduler.config)
        coarse_scheduler.set_timesteps(coarse_steps, device=device)
        coarse_solver = ExactDPMSolver(coarse_scheduler)
        fine_sigmas = ExactDPMSolver(scheduler).sigmas
        assert torch.allclose(
            coarse_solver.sigmas[:-1], fine_sigmas[:-1][::train_stride], rtol=1e-5, atol=0
        ), "the fine sigma grid does not nest the coarse one"
        logger.info(
            "dense sampling: {} fine steps -> {} coarse training pairs per trajectory",
            fine_steps,
            coarse_steps - 1,
        )
    if guidance_scale != 1.0:
        logger.info(
            "guidance {}: the reverse step is driven by the combination, so both branches are "
            "cached and the dataset costs 1.5x. Two forwards per step, so generation is ~2x.",
            guidance_scale,
        )
    meta_base = {
        "guidance_scale": guidance_scale,
        "model_id": str(cfg.model_id),
        "schedule": str(cfg.schedule),
        "solver": "first_order_ode",
        "pairing": "matched_timestep",
        "target_space": "data_prediction",
        # The grid the training pairs live on; for a dense dataset that is the coarse grid.
        "num_inference_steps": int(cfg.num_inference_steps)
        if coarse_solver is None
        else len(coarse_solver.timesteps),
        "duration_s": teacher.duration_s,
        "store_dtype": str(cfg.store_dtype),
        "git_sha": git_sha(),
    }
    if coarse_solver is not None:
        meta_base.update(
            {
                "input_space": "exact_coarse_step",
                "train_stride": train_stride,
                "fine_num_inference_steps": int(cfg.num_inference_steps),
            }
        )

    for offset, record in enumerate(tqdm(records, desc="samples")):
        sample_idx = start + offset
        sample_dir = out_dir / f"sample_{sample_idx:06d}"
        if (sample_dir / "meta.json").exists() and not cfg.overwrite:
            continue

        seed = int(cfg.seed) + sample_idx
        text_audio = teacher.encode_prompt(record["prompt"])
        trajectory, data, grid, uncond = teacher.ode_trajectory(
            text_audio, seed=seed, guidance_scale=guidance_scale, uncond_text_audio=uncond_audio
        )

        # The pair inversion actually needs on this grid: the student sees the cleaner latent at
        # *its own* timestep and must predict the teacher's data prediction at the noisier one,
        # which is what the reverse step consumed. On an EDM grid the matched timestep beats the
        # shifted one 0.0179 to 0.0474 -- see output/sao_schedules/REPORT.md.
        # The last transition ends at sigma = 0, where the reverse step discards the sample and has
        # no inverse, so it is dropped: trajectory[:-1] pairs with data[:-1].
        states = None
        sample_meta = {**meta_base, "sample_idx": sample_idx, "seed": seed}
        if coarse_solver is None:
            trajectory, outputs, timesteps = trajectory[:-1], data[:-1], grid[1:]
            uncond = uncond[:-1] if uncond is not None else None
        else:
            trajectory, outputs, states, gap_rel = dense_coarse_pairs(
                trajectory, data, coarse_solver, train_stride
            )
            timesteps = list(coarse_solver.timesteps)[1:]
            sample_meta["dense_gap_rel_mean"] = float(gap_rel.mean())
            sample_meta["dense_gap_rel_max"] = float(gap_rel.max())
            if offset == 0:
                logger.info(
                    "dense-vs-coarse input gap (the signal this dataset adds): "
                    "rel mean {:.3e}, max {:.3e}",
                    float(gap_rel.mean()),
                    float(gap_rel.max()),
                )

        if offset == 0:
            log_first_sample(teacher, record, trajectory, outputs, timesteps, text_audio)

        save_sample(
            sample_dir,
            trajectory,
            outputs,
            timesteps,
            text_audio,
            record,
            sample_meta,
            store_dtype,
            uncond=uncond,
            states=states,
        )

    logger.success("Wrote trajectories for {} samples to {}", len(records), out_dir)


if __name__ == "__main__":
    main()
