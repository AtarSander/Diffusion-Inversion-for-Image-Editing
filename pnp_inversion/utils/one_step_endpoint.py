"""Shared one-call endpoint inversion for SD1.5 editor pipelines."""
from __future__ import annotations

import torch


@torch.no_grad()
def one_step_endpoint_latent(unet, scheduler, z0: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Map encoded image latent z0 directly to the scheduler's terminal zT.

    The adapter is enabled by the caller for this one UNet call only.  Sampling
    after this function must use the base UNet, exactly as the pure-LoRA
    evaluation paths do for iterative inversion.
    """
    if len(scheduler.timesteps) == 0:
        raise ValueError("Scheduler timesteps must be set before endpoint inversion.")
    device = z0.device
    input_timestep = scheduler.timesteps[-1].to(device=device)
    terminal_timestep = scheduler.timesteps[0].to(device=device)
    dtype = next(unet.parameters()).dtype
    z0 = z0.to(dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    model_input = scheduler.scale_model_input(z0, input_timestep)
    output = unet(model_input, input_timestep, encoder_hidden_states=prompt_embeds)
    eps = output.sample if hasattr(output, "sample") else output[0]
    alphas = scheduler.alphas_cumprod.to(device=device, dtype=z0.dtype)
    alpha = alphas[terminal_timestep].reshape(1, *([1] * (z0.ndim - 1)))
    return alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * eps
