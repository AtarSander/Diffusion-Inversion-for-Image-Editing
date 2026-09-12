# ABOUTME: Pins down why DDPM-inversion reconstructs exactly and ODE inversion cannot: the former
# ABOUTME: solves each step for a free per-step noise, the latter has no free variable to absorb error.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.stable_audio import ExactDPMSolver  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cycle_loss_algebra import FakeCosineScheduler  # noqa: E402


def sde_step(x_t, data_pred, z, sigma_s, sigma_t):
    """One sde-dpmsolver++ first-order reverse step, as diffusers writes it."""
    h = torch.log(sigma_s) - torch.log(sigma_t)
    return (
        (sigma_t / sigma_s * torch.exp(-h)) * x_t
        + (1 - torch.exp(-2.0 * h)) * data_pred
        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * z
    )


def solve_for_z(x_t, x_tm1, data_pred, sigma_s, sigma_t):
    """`get_zs_from_xts`: solve the step for the noise that lands exactly on `x_tm1`."""
    h = torch.log(sigma_s) - torch.log(sigma_t)
    return (
        x_tm1 - (sigma_t / sigma_s * torch.exp(-h)) * x_t - (1 - torch.exp(-2.0 * h)) * data_pred
    ) / (sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)))


@pytest.fixture
def solver():
    return ExactDPMSolver(FakeCosineScheduler())


def test_z_solve_is_an_identity_for_any_prediction(solver):
    """The recovered z replays to x_{t-1} exactly -- even when the prediction is pure garbage.

    This is the whole of DDPM-inversion's exactness: the reconstruction does not depend on the
    model being right, because z absorbs whatever the model got wrong.
    """
    torch.manual_seed(0)
    for index in range(solver.invertible_steps):
        sigma_s, sigma_t = solver.sigmas[index], solver.sigmas[index + 1]
        x_t = torch.randn(2, 4, 8, dtype=torch.float64)
        x_tm1 = torch.randn(2, 4, 8, dtype=torch.float64)
        for prediction in (torch.randn_like(x_t), torch.zeros_like(x_t), 1e3 * torch.randn_like(x_t)):
            z = solve_for_z(x_t, x_tm1, prediction, sigma_s, sigma_t)
            replayed = sde_step(x_t, prediction, z, sigma_s, sigma_t)
            assert torch.allclose(replayed, x_tm1, atol=1e-9), f"step {index}"


def test_ode_inversion_is_exact_only_with_the_unreachable_prediction(solver):
    """ODE inversion has no free variable: a wrong prediction lands in the latent as error.

    The error is exactly -(B/A) times the prediction error -- there is nowhere else for it to go,
    which is why the shifted-denoiser LoRA has to close that gap directly.
    """
    torch.manual_seed(0)
    for index in range(solver.invertible_steps):
        a, b = solver.coefficients(index)
        x_s = torch.randn(2, 4, 8, dtype=torch.float64)
        true_pred = torch.randn(2, 4, 8, dtype=torch.float64)
        x_t = solver.forward(x_s, true_pred, index)

        assert torch.allclose(solver.inverse(x_t, true_pred, index), x_s, atol=1e-10)

        wrong = true_pred + torch.randn_like(true_pred)
        recovered = solver.inverse(x_t, wrong, index)
        assert torch.allclose(recovered - x_s, -(b / a) * (wrong - true_pred), atol=1e-10)
        assert not torch.allclose(recovered, x_s, atol=1e-3), "a wrong prediction must cost accuracy"


def test_ddpm_stores_one_latent_per_step_and_ode_stores_one_total(solver):
    """The codes are not the same size: DDPM-inversion carries a per-step residual, ODE one latent.

    Both reproduce the source, but at T=100 the DDPM code is ~101x larger, so a comparison of the
    two is not a comparison at equal stored information.
    """
    steps, latent = 100, 64 * 1024
    ode_floats = latent                      # the single inverted latent
    ddpm_floats = latent * (steps + 1)       # x_T plus one z per step
    assert ddpm_floats == 101 * ode_floats


def test_independent_marginals_are_not_a_trajectory():
    """`sample_xts_from_x0` draws a fresh noise per timestep, so the xts are not one chain.

    Consecutive latents of a real trajectory share their noise and stay close as sigma shrinks;
    independently drawn ones do not, which is the property the edit-friendly paper exploits.
    """
    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 256, dtype=torch.float64)
    sigmas = torch.tensor([2.0, 1.9], dtype=torch.float64)

    independent = [x0 + torch.randn_like(x0) * s for s in sigmas]
    shared = torch.randn_like(x0)
    trajectory = [x0 + shared * s for s in sigmas]

    gap_independent = (independent[0] - independent[1]).std()
    gap_trajectory = (trajectory[0] - trajectory[1]).std()
    assert gap_trajectory < 0.2 and gap_independent > 2.0, (gap_trajectory, gap_independent)
