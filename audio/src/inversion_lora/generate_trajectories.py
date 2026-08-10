# ABOUTME: Generate AudioLDM2 DDIM latent trajectories from MusicCaps captions and cache the
# ABOUTME: frozen-teacher epsilon targets used to train the shifted-denoiser inversion LoRA.

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

AUDIO_EDITING_CODE = Path(__file__).resolve().parents[2] / "editing/AudioEditingCode/code"
if str(AUDIO_EDITING_CODE) not in sys.path:
    sys.path.insert(0, str(AUDIO_EDITING_CODE))

from models import PipelineWrapper, load_model  # noqa: E402


def git_sha() -> str:
    """Return the current commit SHA so every sample is traceable to its code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
    ).strip()


def load_captions(csv_path: Path, caption_column: str) -> list[dict[str, Any]]:
    """Load MusicCaps caption records, keeping the YouTube id for traceability.

    Args:
        csv_path: Path to `musiccaps-public.csv`.
        caption_column: Column holding the free-text caption.

    Returns:
        One record per row, in file order.
    """
    df = pd.read_csv(csv_path)
    if caption_column not in df.columns:
        raise ValueError(f"Column {caption_column!r} not in {csv_path}: {list(df.columns)}")
    return [
        {"prompt": str(row[caption_column]), "ytid": str(row.get("ytid", ""))}
        for _, row in df.iterrows()
    ]


def latent_height(pipe, audio_length_in_s: float | None) -> int:
    """Compute the AudioLDM2 latent time dimension for a target audio duration.

    Mirrors `AudioLDM2Pipeline` so generated latents match the editing pipelines exactly.
    """
    vocoder_upsample_factor = (
        np.prod(pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate
    )
    if audio_length_in_s is None:
        audio_length_in_s = (
            pipe.unet.config.sample_size * pipe.vae_scale_factor * vocoder_upsample_factor
        )
    height = int(audio_length_in_s / vocoder_upsample_factor)
    if height % pipe.vae_scale_factor != 0:
        height = int(np.ceil(height / pipe.vae_scale_factor)) * pipe.vae_scale_factor
    return height


@torch.no_grad()
def sample_with_trajectory(
    ldm: PipelineWrapper,
    prompt: str,
    guidance_scale: float,
    audio_length_in_s: float | None,
    seed: int,
    save_uncond_target: bool,
) -> dict[str, Any]:
    """Run teacher DDIM denoising, keeping every latent and every conditional epsilon.

    The returned `target_eps[i]` is the frozen teacher's epsilon at `trajectory[i]`, so the
    training pair is (trajectory[i + 1], timesteps[i]) -> target_eps[i].

    Args:
        ldm: Vendored AudioEditing wrapper, already on the right device and scheduler.
        prompt: Caption conditioning the trajectory.
        guidance_scale: CFG scale for the sampling path; 1.0 reproduces conditional-only DDIM.
        audio_length_in_s: Target duration; None uses the model's native length.
        seed: Seed for the initial latent.
        save_uncond_target: Also keep the unconditional epsilon per step.

    Returns:
        Dict with the trajectory, targets, conditioning tensors and timestep values.
    """
    pipe = ldm.model
    device = ldm.device
    scheduler = pipe.scheduler

    hidden, t5_embeds, t5_mask = ldm.encode_text([prompt])
    u_hidden, u_t5_embeds, u_t5_mask = ldm.encode_text([""], negative=True)

    needs_uncond = guidance_scale != 1.0 or save_uncond_target

    height = latent_height(pipe, audio_length_in_s)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        dtype=pipe.unet.dtype,
        device=torch.device(device),
        generator=torch.Generator(device=device).manual_seed(seed),
        latents=None,
    )

    trajectory = [latents.detach().cpu()]
    target_eps: list[torch.Tensor] = []
    uncond_eps: list[torch.Tensor] = []
    timestep_values: list[int] = []

    for t in tqdm(scheduler.timesteps, desc="denoising", leave=False):
        model_input = scheduler.scale_model_input(latents, t)

        eps_cond = ldm.unet_forward(
            model_input,
            timestep=t,
            encoder_hidden_states=hidden,
            class_labels=t5_embeds,
            encoder_attention_mask=t5_mask,
        )[0].sample

        if needs_uncond:
            eps_uncond = ldm.unet_forward(
                model_input,
                timestep=t,
                encoder_hidden_states=u_hidden,
                class_labels=u_t5_embeds,
                encoder_attention_mask=u_t5_mask,
            )[0].sample
        else:
            eps_uncond = None

        target_eps.append(eps_cond.detach().cpu())
        if save_uncond_target:
            uncond_eps.append(eps_uncond.detach().cpu())

        if guidance_scale == 1.0:
            eps_step = eps_cond
        else:
            eps_step = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        latents = scheduler.step(
            model_output=eps_step, timestep=t, sample=latents, return_dict=True
        ).prev_sample

        trajectory.append(latents.detach().cpu())
        timestep_values.append(int(t.item()) if hasattr(t, "item") else int(t))

    result = {
        "final_latent": latents,
        "trajectory": torch.cat(trajectory, dim=0),
        "target_eps": torch.cat(target_eps, dim=0),
        "timesteps": timestep_values,
        "conditioning": {
            "generated_prompt_embeds": hidden.detach().cpu()[0],
            "t5_prompt_embeds": t5_embeds.detach().cpu()[0],
            "t5_attention_mask": t5_mask.detach().cpu()[0],
        },
    }
    if save_uncond_target:
        result["uncond_eps"] = torch.cat(uncond_eps, dim=0)
    return result


def save_sample(
    ldm: PipelineWrapper,
    sample: dict[str, Any],
    record: dict[str, Any],
    sample_idx: int,
    seed: int,
    cfg: DictConfig,
    out_dir: Path,
) -> None:
    """Persist one trajectory sample directory."""
    sample_dir = out_dir / f"sample_{sample_idx:06d}"
    (sample_dir / "latents").mkdir(parents=True, exist_ok=True)
    (sample_dir / "targets").mkdir(parents=True, exist_ok=True)

    store_dtype = getattr(torch, str(cfg.store_dtype))
    trajectory = sample["trajectory"].to(store_dtype)
    target_eps = sample["target_eps"].to(store_dtype)

    assert trajectory.shape[0] == target_eps.shape[0] + 1, (
        f"trajectory has {trajectory.shape[0]} latents but {target_eps.shape[0]} targets"
    )
    assert len(sample["timesteps"]) == target_eps.shape[0], (
        f"{len(sample['timesteps'])} timesteps vs {target_eps.shape[0]} targets"
    )

    torch.save(trajectory, sample_dir / "latents/trajectory.pt")
    torch.save(target_eps, sample_dir / "targets/target_eps.pt")
    torch.save(sample["conditioning"], sample_dir / "conditioning.pt")
    if "uncond_eps" in sample:
        torch.save(sample["uncond_eps"].to(store_dtype), sample_dir / "targets/uncond_eps.pt")

    with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
        json.dump(sample["timesteps"], f)
    with (sample_dir / "prompt.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if cfg.save_final_audio:
        mel = ldm.vae_decode(sample["final_latent"])
        if mel.dim() < 4:
            mel = mel[None]
        audio = ldm.decode_to_mel(mel)
        import torchaudio

        torchaudio.save(
            str(sample_dir / "final.wav"), audio.detach().cpu().float(), sample_rate=ldm.get_sr()
        )

    meta = {
        "sample_idx": sample_idx,
        "seed": seed,
        "model_id": cfg.model_id,
        "num_inference_steps": int(cfg.num_inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "audio_length_in_s": cfg.audio_length_in_s,
        "trajectory_length": int(trajectory.shape[0]),
        "num_transitions": int(target_eps.shape[0]),
        "latent_shape": list(trajectory.shape[1:]),
        "t5_seq_len": int(sample["conditioning"]["t5_prompt_embeds"].shape[0]),
        "store_dtype": str(cfg.store_dtype),
        "git_sha": git_sha(),
    }
    # meta.json is the completion sentinel, so write it last and atomically: a crash between
    # tensor writes must not leave a directory that resume treats as finished.
    tmp_meta = sample_dir / "meta.json.tmp"
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(sample_dir / "meta.json")


@hydra.main(config_path="../../config", config_name="generate_trajectories", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate and cache AudioLDM2 trajectories for inversion-LoRA training."""
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
    ldm = load_model(
        str(cfg.model_id), device, int(cfg.num_inference_steps), edit_method="ddim"
    )
    ldm.model.unet.eval()
    logger.info(
        "scheduler={} prediction_type={} timesteps={}",
        type(ldm.model.scheduler).__name__,
        ldm.model.scheduler.config.prediction_type,
        len(ldm.model.scheduler.timesteps),
    )

    for offset, record in enumerate(tqdm(records, desc="samples")):
        sample_idx = start + offset
        sample_dir = out_dir / f"sample_{sample_idx:06d}"
        if (sample_dir / "meta.json").exists() and not cfg.overwrite:
            continue

        seed = int(cfg.seed) + sample_idx
        sample = sample_with_trajectory(
            ldm=ldm,
            prompt=record["prompt"],
            guidance_scale=float(cfg.guidance_scale),
            audio_length_in_s=cfg.audio_length_in_s,
            seed=seed,
            save_uncond_target=bool(cfg.save_uncond_target),
        )

        if offset == 0:
            logger.info("First prompt: {!r}", record["prompt"])
            logger.info(
                "trajectory={} target_eps={} t5_embeds={} timesteps[:3]={}",
                tuple(sample["trajectory"].shape),
                tuple(sample["target_eps"].shape),
                tuple(sample["conditioning"]["t5_prompt_embeds"].shape),
                sample["timesteps"][:3],
            )

        save_sample(ldm, sample, record, sample_idx, seed, cfg, out_dir)

    logger.success("Wrote trajectories for {} samples to {}", len(records), out_dir)


if __name__ == "__main__":
    main()
