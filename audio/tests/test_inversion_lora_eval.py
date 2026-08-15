# ABOUTME: Known-input tests for the quartile loss bucketing, the mel reconstruction metrics and
# ABOUTME: the noise distribution checks used by the inversion-LoRA training eval.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.noise_metrics import (  # noqa: E402
    kl_div_per_dim,
    kl_div_scalar,
    top_k_corr_in_patches,
)
from src.inversion_lora.reconstruct import mel_metrics  # noqa: E402
from src.inversion_lora.train import NUM_QUARTILES, timestep_quartile  # noqa: E402


def test_quartiles_split_the_schedule_noisiest_first():
    """Quartile 0 must be the noisiest end, so q1 reads as 'the first 25% of denoising steps'."""
    # Boundaries are half-open on the noisy side: t=750 opens quartile 1, t=250 opens quartile 3.
    timesteps = torch.tensor([999, 800, 750, 600, 500, 300, 250, 100, 0])
    quartiles = timestep_quartile(timesteps, num_train_timesteps=1000)
    assert quartiles.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_quartiles_cover_a_real_ddim_grid_evenly():
    """The 200-step grid this model uses must land 50 steps in each bucket."""
    timesteps = torch.arange(996, 0, -5)
    assert len(timesteps) == 200
    counts = torch.bincount(
        timestep_quartile(timesteps, num_train_timesteps=1000), minlength=NUM_QUARTILES
    )
    assert counts.tolist() == [50, 50, 50, 50]


def test_quartile_index_stays_in_range_at_the_boundary():
    assert timestep_quartile(torch.tensor([0]), 1000).item() == NUM_QUARTILES - 1
    assert timestep_quartile(torch.tensor([1000]), 1000).item() == 0


def test_identical_mel_is_a_perfect_reconstruction():
    """The metric that says 'nothing changed' has to agree with itself."""
    mel = torch.randn(2, 1, 64, 32)
    scores = mel_metrics(mel, mel)
    assert scores["mel_mse"] == pytest.approx(0.0, abs=1e-12)
    assert scores["mel_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert scores["mel_psnr"] > 100  # skimage returns inf-free but very large for exact matches


def test_mel_metrics_degrade_with_noise():
    mel = torch.randn(2, 1, 64, 32)
    slightly_off = mel + 0.05 * torch.randn_like(mel)
    badly_off = mel + 1.0 * torch.randn_like(mel)
    good = mel_metrics(mel, slightly_off)
    bad = mel_metrics(mel, badly_off)
    assert good["mel_mse"] < bad["mel_mse"]
    assert good["mel_ssim"] > bad["mel_ssim"]
    assert good["mel_psnr"] > bad["mel_psnr"]


def test_kl_of_a_distribution_with_itself_is_zero():
    sample = torch.randn(16, 4, 8, 8)
    assert kl_div_scalar(sample, sample) == pytest.approx(0.0, abs=1e-6)
    assert kl_div_per_dim(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_kl_grows_when_the_scale_is_wrong():
    """A latent that is not unit-variance is exactly what these metrics must catch."""
    reference = torch.randn(64, 4, 8, 8)
    assert kl_div_scalar(reference, 2.0 * reference) > kl_div_scalar(reference, 1.1 * reference)
    assert kl_div_per_dim(reference, 2.0 * reference) > kl_div_per_dim(reference, 1.1 * reference)


def test_correlated_latents_score_above_iid_noise():
    """The point of the correlation metric: structure left in the inverted latent shows up."""
    torch.manual_seed(0)
    iid = torch.randn(32, 4, 16, 16)
    # Every position in a patch driven by one shared per-example factor: maximally correlated.
    structured = torch.randn(32, 1, 1, 1).expand(32, 4, 16, 16).contiguous()
    assert top_k_corr_in_patches(structured)["mean"] > top_k_corr_in_patches(iid)["mean"]


def test_correlation_needs_more_than_one_example():
    with pytest.raises(AssertionError, match="at least 2 examples"):
        top_k_corr_in_patches(torch.randn(1, 4, 16, 16))
