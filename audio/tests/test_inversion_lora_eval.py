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
from src.inversion_lora.dataset import transitions_below_timestep  # noqa: E402
from src.inversion_lora.train import band_labels, timestep_band  # noqa: E402

DDIM_GRID = torch.arange(996, 0, -5)  # the 200-step grid this model trains on


def test_bands_split_the_schedule_noisiest_first():
    """Band 0 must be the noisiest end, so it reads as 'the first 25% of denoising steps'."""
    # Boundaries are half-open on the noisy side: t=750 opens band 1, t=250 opens band 3.
    timesteps = torch.tensor([999, 800, 750, 600, 500, 300, 250, 100, 0])
    bands = timestep_band(timesteps, band_top=1000, num_bands=4)
    assert bands.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_bands_cover_a_real_ddim_grid_evenly():
    """The 200-step grid must land 50 steps in each quarter and 10 in each fifth of the last."""
    assert len(DDIM_GRID) == 200
    quarters = torch.bincount(timestep_band(DDIM_GRID, 1000, 4), minlength=4)
    assert quarters.tolist() == [50, 50, 50, 50]

    tail = DDIM_GRID[DDIM_GRID <= 250]
    assert len(tail) == 50
    fifths = torch.bincount(timestep_band(tail, 250, 5), minlength=5)
    assert fifths.tolist() == [10, 10, 10, 10, 10]


def test_band_index_stays_in_range_at_the_boundary():
    assert timestep_band(torch.tensor([0]), 1000, 4).item() == 3
    assert timestep_band(torch.tensor([1000]), 1000, 4).item() == 0


def test_bands_that_do_not_cover_the_data_are_rejected():
    """Setting the fine bands without restricting the data must fail, not silently mislabel."""
    with pytest.raises(AssertionError, match="above band_top"):
        timestep_band(DDIM_GRID, band_top=250, num_bands=5)


def test_band_names_report_their_share_of_the_full_schedule():
    assert band_labels(1000, 4, 1000) == ["q100_75", "q75_50", "q50_25", "q25_00"]
    assert band_labels(250, 5, 1000) == ["q25_20", "q20_15", "q15_10", "q10_05", "q05_00"]


def test_restricting_the_dataset_keeps_exactly_the_cleanest_quarter():
    """The filter must select by timestep, not by position, across trajectory boundaries."""

    class FakeDataset:
        samples = [
            {"timesteps": DDIM_GRID.tolist(), "num_transitions": 200},
            {"timesteps": DDIM_GRID.tolist(), "num_transitions": 200},
        ]

    keep = transitions_below_timestep(FakeDataset(), 250)
    assert len(keep) == 100
    assert keep[:3] == [150, 151, 152] and keep[50:53] == [350, 351, 352]
    assert max(DDIM_GRID[i % 200] for i in keep) == 246


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
