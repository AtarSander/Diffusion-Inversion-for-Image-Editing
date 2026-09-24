"""Matched SDXL DDIM/TADA inversion and reconstruction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import decode_latent_to_pil, encode_prompt_sdxl
from diff_inversion.eval.invert_sdxl import _load_run_config, _load_tensor, _read_json
from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl, reconstruct_latent_sdxl
from diff_inversion.utils import make_pipe


def exact_ddim_inverse_step(
    scheduler: DDIMInverseScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    sample: torch.Tensor,
) -> torch.Tensor:
    """TADA's DDIM inverse update with source/target noise levels aligned exactly."""
    if scheduler.num_inference_steps is None:
        raise ValueError("Call scheduler.set_timesteps before exact_ddim_inverse_step.")
    if str(scheduler.config.prediction_type) != "epsilon":
        raise ValueError("The matched TADA experiment requires epsilon prediction.")

    target_timestep = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
    step_ratio = int(scheduler.config.num_train_timesteps) // scheduler.num_inference_steps
    source_timestep = min(
        target_timestep - step_ratio,
        int(scheduler.config.num_train_timesteps) - 1,
    )
    alpha_source = (
        scheduler.alphas_cumprod[source_timestep]
        if source_timestep >= 0
        else scheduler.initial_alpha_cumprod
    )
    alpha_target = scheduler.alphas_cumprod[target_timestep]
    alpha_source = alpha_source.to(device=sample.device, dtype=sample.dtype)
    alpha_target = alpha_target.to(device=sample.device, dtype=sample.dtype)

    pred_original = (
        sample - (1 - alpha_source).sqrt() * model_output
    ) / alpha_source.sqrt()
    if bool(scheduler.config.clip_sample):
        clip_range = float(scheduler.config.clip_sample_range)
        pred_original = pred_original.clamp(-clip_range, clip_range)
    return alpha_target.sqrt() * pred_original + (1 - alpha_target).sqrt() * model_output


def forward_diffuse(
    scheduler: DDIMInverseScheduler,
    clean_latent: torch.Tensor,
    timestep: torch.Tensor | int,
    fixed_noise: torch.Tensor,
) -> torch.Tensor:
    """Closed-form forward diffusion used to replace the first TADA inverse steps."""
    timestep_value = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
    alpha_bar = scheduler.alphas_cumprod[timestep_value].to(
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    return alpha_bar.sqrt() * clean_latent + (1 - alpha_bar).sqrt() * fixed_noise


@torch.no_grad()
def invert_latent(
    *,
    pipe,
    clean_latent: torch.Tensor,
    conditioning: dict[str, torch.Tensor],
    scheduler_config,
    num_inference_steps: int,
    guidance_scale: float,
    forward_steps: int,
    fixed_noise: torch.Tensor | None,
    progress_desc: str | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Run exact DDIM inversion, optionally replacing its first steps with TADA."""
    scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    pipe.scheduler = scheduler
    timesteps = scheduler.timesteps
    latents = clean_latent.to(device=pipe.device, dtype=pipe.unet.dtype)
    used_timesteps: list[int] = []

    if forward_steps:
        if not 0 < forward_steps < len(timesteps):
            raise ValueError(
                f"forward_steps must be in [1, {len(timesteps) - 1}], got {forward_steps}"
            )
        if fixed_noise is None:
            raise ValueError("fixed_noise is required when forward_steps > 0")
        forward_timestep = timesteps[forward_steps - 1]
        latents = forward_diffuse(scheduler, latents, forward_timestep, fixed_noise)
        used_timesteps.append(int(forward_timestep.item()))
        timesteps = timesteps[forward_steps:]

    for timestep in tqdm(
        timesteps,
        desc=progress_desc,
        leave=False,
        disable=progress_desc is None,
    ):
        noise_pred = predict_noise_sdxl(
            pipe=pipe,
            latents=latents,
            timestep=timestep,
            cond=conditioning,
            guidance_scale=guidance_scale,
        )
        latents = exact_ddim_inverse_step(scheduler, noise_pred, timestep, latents)
        used_timesteps.append(int(timestep.item()))
    return latents.detach().cpu(), used_timesteps


def _sample_indices(cfg: DictConfig) -> range:
    start = int(cfg.start_index)
    count = int(cfg.num_samples)
    if start < 0 or count < 1:
        raise ValueError(f"Invalid sample range: start_index={start}, num_samples={count}")
    return range(start, start + count)


def _variant_dicts(cfg: DictConfig) -> list[dict[str, Any]]:
    variants = OmegaConf.to_container(cfg.variants, resolve=True)
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a non-empty list")
    names = [str(variant["name"]) for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError(f"Variant names must be unique, got {names}")
    return variants


@hydra.main(config_path="../../config", config_name="eval/tada_sdxl", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Matched TADA config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = Path(to_absolute_path(str(cfg.input_dir))).resolve()
    run_cfg = _load_run_config(input_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if run_cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("The matched TADA experiment requires CUDA.")
    pipe = make_pipe(run_cfg.model, device)
    scheduler_config = pipe.scheduler.config
    variants = _variant_dicts(cfg)

    for sample_idx in tqdm(_sample_indices(cfg), desc="Matched TADA samples"):
        sample_dir = input_dir / str(cfg.sample_dir_template).format(sample_idx=sample_idx)
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing generated sample: {sample_dir}")
        prompt_record = _read_json(sample_dir / "prompt.json")
        prompt = str(prompt_record["prompt"])
        clean_latent = _load_tensor(sample_dir / "latents" / "final_latent.pt").to(
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )
        conditioning = encode_prompt_sdxl(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=str(run_cfg.negative_prompt),
            height=int(run_cfg.model.height),
            width=int(run_cfg.model.width),
        )
        generator = torch.Generator(device=pipe.device).manual_seed(
            int(cfg.forward_seed) + sample_idx
        )
        fixed_noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        )
        sample_metadata: dict[str, Any] = {
            "sample_idx": sample_idx,
            "forward_seed": int(cfg.forward_seed) + sample_idx,
            "variants": {},
        }

        for variant in variants:
            name = str(variant["name"])
            forward_steps = int(variant["forward_steps"])
            noise_path = sample_dir / str(variant["noise_name"])
            image_path = sample_dir / str(variant["image_name"])
            overwrite = bool(cfg.overwrite)

            if noise_path.exists() and not overwrite:
                inverted_noise = _load_tensor(noise_path)
                inversion_timesteps: list[int] = []
                logger.info("Reusing {}", noise_path)
            else:
                inverted_noise, inversion_timesteps = invert_latent(
                    pipe=pipe,
                    clean_latent=clean_latent,
                    conditioning=conditioning,
                    scheduler_config=scheduler_config,
                    num_inference_steps=int(run_cfg.model.num_inference_steps),
                    guidance_scale=float(run_cfg.model.guidance_scale),
                    forward_steps=forward_steps,
                    fixed_noise=fixed_noise if forward_steps else None,
                    progress_desc=f"{sample_dir.name} {name} inversion",
                )
                torch.save(inverted_noise, noise_path)

            if image_path.exists() and not overwrite:
                logger.info("Reusing {}", image_path)
                reconstruction_timesteps: list[int] = []
            else:
                pipe.scheduler = DDIMScheduler.from_config(scheduler_config)
                reconstructed_latent, reconstruction_timesteps = reconstruct_latent_sdxl(
                    pipe=pipe,
                    noise_latent=inverted_noise,
                    cond=conditioning,
                    num_inference_steps=int(run_cfg.model.num_inference_steps),
                    guidance_scale=float(run_cfg.model.guidance_scale),
                    progress_desc=f"{sample_dir.name} {name} reconstruction",
                )
                image = decode_latent_to_pil(
                    pipe,
                    reconstructed_latent.to(device=pipe.device),
                )
                image.save(image_path)

            sample_metadata["variants"][name] = {
                "forward_steps": forward_steps,
                "forward_fraction": forward_steps / int(run_cfg.model.num_inference_steps),
                "inverted_noise": noise_path.name,
                "reconstructed_image": image_path.name,
                "inversion_timesteps": inversion_timesteps,
                "reconstruction_timesteps": reconstruction_timesteps,
            }

        with (sample_dir / "tada_matched.json").open("w", encoding="utf-8") as handle:
            json.dump(sample_metadata, handle, indent=2)

    logger.success(
        "Finished matched TADA variants for samples [{}:{})",
        int(cfg.start_index),
        int(cfg.start_index) + int(cfg.num_samples),
    )


if __name__ == "__main__":
    main()
