import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

from diff_inversion.eval.tada_sdxl import exact_ddim_inverse_step, forward_diffuse


def test_forward_diffuse_matches_closed_form():
    scheduler = DDIMInverseScheduler.from_config(DDIMScheduler().config)
    scheduler.set_timesteps(50)
    clean = torch.full((1, 4, 2, 2), 0.25)
    noise = torch.full_like(clean, -0.5)
    timestep = scheduler.timesteps[1]
    alpha = scheduler.alphas_cumprod[int(timestep)].to(clean)
    expected = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    torch.testing.assert_close(forward_diffuse(scheduler, clean, timestep, noise), expected)


def test_first_exact_inverse_step_targets_first_scheduler_noise_level():
    scheduler = DDIMInverseScheduler.from_config(DDIMScheduler().config)
    scheduler.set_timesteps(50)
    clean = torch.randn(1, 4, 2, 2)
    eps = torch.randn_like(clean)
    timestep = scheduler.timesteps[0]
    actual = exact_ddim_inverse_step(scheduler, eps, timestep, clean)
    alpha = scheduler.alphas_cumprod[int(timestep)].to(clean)
    expected = alpha.sqrt() * clean + (1 - alpha).sqrt() * eps
    torch.testing.assert_close(actual, expected)
