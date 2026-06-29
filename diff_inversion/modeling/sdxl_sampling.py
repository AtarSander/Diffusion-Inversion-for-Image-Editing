"""Shared SDXL sampling helpers used by evaluation and training previews."""

from __future__ import annotations

import torch
from diffusers import DDIMInverseScheduler
from tqdm import tqdm


@torch.no_grad()
def predict_noise_sdxl(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: dict[str, torch.Tensor],
    guidance_scale: float,
) -> torch.Tensor:
    """Predict CFG-combined SDXL noise for one latent batch."""
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    encoder_hidden_states = torch.cat(
        [cond["negative_prompt_embeds"], cond["prompt_embeds"]],
        dim=0,
    )
    unet_kwargs = {}
    if "pooled_prompt_embeds" in cond and "add_time_ids" in cond:
        text_embeds = torch.cat(
            [cond["negative_pooled_prompt_embeds"], cond["pooled_prompt_embeds"]],
            dim=0,
        )
        time_ids = cond["add_time_ids"].repeat(2, 1)
        unet_kwargs["added_cond_kwargs"] = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }

    noise_pred = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
        **unet_kwargs,
    )[0]

    noise_uncond, noise_text = noise_pred.chunk(2)
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)


@torch.no_grad()
def invert_latent_sdxl(
    *,
    pipe,
    final_latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    scheduler_config,
    num_inference_steps: int,
    guidance_scale: float,
    progress_desc: str | None = None,
) -> torch.Tensor:
    inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    pipe.scheduler = inverse_scheduler
    inverse_scheduler.set_timesteps(num_inference_steps, device=pipe.device)

    latents = final_latent.to(device=pipe.device, dtype=pipe.unet.dtype)
    for timestep in tqdm(
        inverse_scheduler.timesteps,
        desc=progress_desc,
        leave=False,
        disable=progress_desc is None,
    ):
        noise_pred = predict_noise_sdxl(
            pipe=pipe,
            latents=latents,
            timestep=timestep,
            cond=cond,
            guidance_scale=guidance_scale,
        )
        latents = inverse_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            return_dict=True,
        ).prev_sample
    return latents.detach()


@torch.no_grad()
def reconstruct_latent_sdxl(
    *,
    pipe,
    noise_latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    num_inference_steps: int,
    guidance_scale: float,
    progress_desc: str | None = None,
) -> tuple[torch.Tensor, list[int]]:
    if noise_latent.ndim == 3:
        noise_latent = noise_latent.unsqueeze(0)
    if noise_latent.ndim != 4 or noise_latent.shape[0] != 1:
        raise ValueError(
            "Expected inverted noise with shape [1,C,H,W] or [C,H,W], "
            f"got {tuple(noise_latent.shape)}"
        )

    latents = noise_latent.to(device=pipe.device, dtype=pipe.unet.dtype)
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timestep_values = []

    for timestep in tqdm(
        pipe.scheduler.timesteps,
        desc=progress_desc,
        leave=False,
        disable=progress_desc is None,
    ):
        timestep_values.append(
            int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
        )
        noise_pred = predict_noise_sdxl(
            pipe=pipe,
            latents=latents,
            timestep=timestep,
            cond=cond,
            guidance_scale=guidance_scale,
        )
        latents = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            return_dict=True,
        ).prev_sample

    return latents.detach().cpu(), timestep_values
