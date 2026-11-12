import torch
from diffusers import DDIMScheduler

from diff_inversion.eval.sd15_cycle_metrics import (
    _generation_step,
    _inverse_step,
    _per_item_metrics,
)


def _scheduler() -> DDIMScheduler:
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(50)
    return scheduler


def test_cycle_inverse_and_generation_are_exact_for_the_same_epsilon() -> None:
    scheduler = _scheduler()
    timesteps = scheduler.timesteps[torch.tensor([0, 17, 49])]
    x_clean = torch.randn(3, 4, 8, 8)
    eps = torch.randn_like(x_clean)

    x_noisy = _inverse_step(scheduler, x_clean, eps, timesteps)
    reconstructed = _generation_step(scheduler, x_noisy, eps, timesteps)

    torch.testing.assert_close(reconstructed, x_clean, rtol=1e-5, atol=2e-6)


def test_cycle_metric_is_zero_for_exact_reconstruction() -> None:
    latent = torch.randn(2, 4, 8, 8)
    metrics = _per_item_metrics(latent, latent)

    for metric in metrics.values():
        torch.testing.assert_close(metric, torch.zeros_like(metric))
