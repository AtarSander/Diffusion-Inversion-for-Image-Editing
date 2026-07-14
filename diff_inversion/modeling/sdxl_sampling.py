"""Shared SDXL sampling helpers used by evaluation and training previews."""

from __future__ import annotations

import torch
from diffusers import DDIMInverseScheduler
from tqdm import tqdm

from diff_inversion.modeling.cfg_temb import cfg_temb_context


def _set_active_adapter(pipe, adapter_name: str) -> None:
    switched = False
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
            switched = True
    if not switched:
        raise AttributeError("No injected LoRA adapter layers expose set_adapter().")


def _predict_noise_sdxl_single_branch(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor | None,
    add_time_ids: torch.Tensor | None,
) -> torch.Tensor:
    latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
    unet_kwargs = {}
    if pooled_prompt_embeds is not None:
        if add_time_ids is None:
            raise KeyError("add_time_ids are required for SDXL pooled prompt embeddings.")
        unet_kwargs["added_cond_kwargs"] = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }
    return pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
        **unet_kwargs,
    )[0]


def _guidance_scale_tensor(
    guidance_scale: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if torch.is_tensor(guidance_scale):
        values = guidance_scale.to(device=device, dtype=torch.float32).flatten()
    else:
        values = torch.tensor([float(guidance_scale)], device=device, dtype=torch.float32)
    if values.numel() == 1:
        values = values.expand(batch_size)
    if values.numel() != batch_size:
        raise ValueError(
            "guidance_scale batch shape does not match latent batch shape: "
            f"got {values.numel()} values for batch_size={batch_size}."
        )
    return values


def _combine_cfg_predictions(
    noise_uncond: torch.Tensor,
    noise_cond: torch.Tensor,
    guidance_scale: torch.Tensor,
) -> torch.Tensor:
    guidance_scale = guidance_scale.to(
        device=noise_uncond.device,
        dtype=noise_uncond.dtype,
    ).reshape(noise_uncond.shape[0], *([1] * (noise_uncond.ndim - 1)))
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


@torch.no_grad()
def predict_noise_sdxl_branches(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: dict[str, torch.Tensor],
    guidance_scale: float | torch.Tensor,
    lora_branch_adapter_names: tuple[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict unconditional, conditional, and CFG-combined SDXL noise."""
    guidance_values = _guidance_scale_tensor(
        guidance_scale,
        batch_size=latents.shape[0],
        device=latents.device,
    )
    if lora_branch_adapter_names is not None:
        unconditional_adapter_name, conditional_adapter_name = lora_branch_adapter_names
        pooled_prompt_embeds = cond.get("pooled_prompt_embeds")
        negative_pooled_prompt_embeds = cond.get("negative_pooled_prompt_embeds")
        add_time_ids = cond.get("add_time_ids")

        with cfg_temb_context(pipe.unet, guidance_values):
            _set_active_adapter(pipe, unconditional_adapter_name)
            noise_uncond = _predict_noise_sdxl_single_branch(
                pipe,
                latents,
                timestep,
                cond["negative_prompt_embeds"],
                negative_pooled_prompt_embeds,
                add_time_ids,
            )

            _set_active_adapter(pipe, conditional_adapter_name)
            noise_text = _predict_noise_sdxl_single_branch(
                pipe,
                latents,
                timestep,
                cond["prompt_embeds"],
                pooled_prompt_embeds,
                add_time_ids,
            )

        noise_cfg = _combine_cfg_predictions(noise_uncond, noise_text, guidance_values)
        return noise_uncond, noise_text, noise_cfg

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

    with cfg_temb_context(pipe.unet, guidance_values):
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            **unet_kwargs,
        )[0]

    noise_uncond, noise_text = noise_pred.chunk(2)
    noise_cfg = _combine_cfg_predictions(noise_uncond, noise_text, guidance_values)
    return noise_uncond, noise_text, noise_cfg


@torch.no_grad()
def predict_noise_sdxl(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: dict[str, torch.Tensor],
    guidance_scale: float | torch.Tensor,
    lora_branch_adapter_names: tuple[str, str] | None = None,
) -> torch.Tensor:
    """Predict CFG-combined SDXL noise for one latent batch."""
    _, _, noise_cfg = predict_noise_sdxl_branches(
        pipe=pipe,
        latents=latents,
        timestep=timestep,
        cond=cond,
        guidance_scale=guidance_scale,
        lora_branch_adapter_names=lora_branch_adapter_names,
    )
    return noise_cfg


@torch.no_grad()
def invert_latent_sdxl(
    *,
    pipe,
    final_latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    scheduler_config,
    num_inference_steps: int,
    guidance_scale: float,
    lora_branch_adapter_names: tuple[str, str] | None = None,
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
            lora_branch_adapter_names=lora_branch_adapter_names,
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
    lora_branch_adapter_names: tuple[str, str] | None = None,
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
            lora_branch_adapter_names=lora_branch_adapter_names,
        )
        latents = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            return_dict=True,
        ).prev_sample

    return latents.detach().cpu(), timestep_values
