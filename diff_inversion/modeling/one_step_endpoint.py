"""One-call SD1.5 endpoint inversion helpers."""
from __future__ import annotations

import torch


def _expand_timestep(timestep: torch.Tensor | int, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(timestep, device=device, dtype=torch.long).flatten()
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(f"Timestep batch has {value.numel()} values for batch size {batch_size}.")
    return value


def endpoint_latent_from_noise(
    z0: torch.Tensor,
    eps: torch.Tensor,
    terminal_timestep: torch.Tensor | int,
    scheduler,
) -> torch.Tensor:
    """Map clean z0 to the cached terminal DDIM latent with one predicted noise field."""
    terminal = _expand_timestep(terminal_timestep, z0.shape[0], z0.device)
    alphas = scheduler.alphas_cumprod.to(device=z0.device, dtype=z0.dtype)
    alpha = alphas[terminal].reshape(-1, *([1] * (z0.ndim - 1)))
    return alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * eps


def endpoint_noise_target(
    z0: torch.Tensor,
    z_terminal: torch.Tensor,
    terminal_timestep: torch.Tensor | int,
    scheduler,
) -> torch.Tensor:
    """Return the global noise field whose closed-form forward step reaches z_terminal."""
    terminal = _expand_timestep(terminal_timestep, z0.shape[0], z0.device)
    alphas = scheduler.alphas_cumprod.to(device=z0.device, dtype=z0.dtype)
    alpha = alphas[terminal].reshape(-1, *([1] * (z0.ndim - 1)))
    sigma = (1.0 - alpha).sqrt()
    if bool(torch.any(sigma == 0).item()):
        raise ValueError("Terminal timestep has zero noise scale.")
    return (z_terminal - alpha.sqrt() * z0) / sigma


@torch.no_grad()
def invert_one_step(
    *,
    unet,
    scheduler,
    z0: torch.Tensor,
    prompt_embeds: torch.Tensor,
    input_timestep: torch.Tensor | int,
    terminal_timestep: torch.Tensor | int,
    added_cond_kwargs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict z_terminal in exactly one conditional UNet call."""
    input_t = _expand_timestep(input_timestep, z0.shape[0], z0.device)
    model_input = scheduler.scale_model_input(z0, input_t)
    kwargs = {} if added_cond_kwargs is None else {"added_cond_kwargs": added_cond_kwargs}
    eps = unet(model_input, input_t, encoder_hidden_states=prompt_embeds, return_dict=False, **kwargs)[0]
    return endpoint_latent_from_noise(z0, eps, terminal_timestep, scheduler)
