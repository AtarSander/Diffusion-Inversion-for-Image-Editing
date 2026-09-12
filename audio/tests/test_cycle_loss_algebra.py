# ABOUTME: Tests the algebra the cycle loss rests on: exact step inversion, batched coefficients,
# ABOUTME: and that the round-trip residual collapses to B * (student - teacher) with no network.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.stable_audio import ExactDPMSolver  # noqa: E402


class FakeCosineScheduler:
    """A geometric sigma grid with the EDM alpha convention, the shape Stable Audio's has."""

    def __init__(self, num_steps: int = 10, sigma_max: float = 100.0, sigma_min: float = 0.3):
        ratio = (sigma_min / sigma_max) ** (1.0 / (num_steps - 1))
        sigmas = torch.tensor([sigma_max * ratio**i for i in range(num_steps)], dtype=torch.float64)
        # final_sigmas_type="zero": the last transition lands on sigma = 0 and has no inverse.
        self.sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])
        self.timesteps = (sigmas.atan() * 2.0 / torch.pi).tolist()

    def _sigma_to_alpha_sigma_t(self, sigma):
        return torch.ones_like(sigma), sigma


@pytest.fixture
def solver() -> ExactDPMSolver:
    return ExactDPMSolver(FakeCosineScheduler())


def test_geometric_grid_gives_constant_coefficients(solver):
    """A log-uniform sigma grid makes every step share one (A, B), with A + B == 1."""
    pairs = [solver.coefficients(i) for i in range(solver.invertible_steps)]
    first_a, first_b = pairs[0]
    for a, b in pairs:
        assert torch.isclose(a, first_a), (a, first_a)
        assert torch.isclose(b, first_b), (b, first_b)
        assert torch.isclose(a + b, torch.ones_like(a)), "EDM alpha=1 implies A + B == 1"


def test_index_for_round_trips(solver):
    """Grid timesteps map back to the indices they came from."""
    want = torch.arange(solver.invertible_steps)
    timesteps = torch.tensor([solver.timesteps[int(i)] for i in want], dtype=torch.float32)
    assert torch.equal(solver.index_for(timesteps), want)


def test_index_for_rejects_off_grid_timesteps(solver):
    """A timestep the solver does not own is a dataset/solver mismatch, not something to round."""
    with pytest.raises(AssertionError, match="not on this solver's grid"):
        solver.index_for(torch.tensor([0.123456]))


def test_coefficients_batch_matches_scalar(solver):
    """The batched lookup agrees with the per-index one and broadcasts as [B, 1, 1]."""
    index = torch.tensor([0, 2, 2, 5])
    a, b = solver.coefficients_batch(index)
    assert a.shape == (4, 1, 1) and b.shape == (4, 1, 1), (a.shape, b.shape)
    for row, i in enumerate(index):
        want_a, want_b = solver.coefficients(int(i))
        assert torch.isclose(a[row, 0, 0], want_a.double())
        assert torch.isclose(b[row, 0, 0], want_b.double())


def test_inverse_undoes_forward(solver):
    """Given the prediction the step consumed, the inverse is exact arithmetic."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    data = torch.randn(2, 4, 8, dtype=torch.float64)
    for index in range(solver.invertible_steps):
        recovered = solver.inverse(solver.forward(x, data, index), data, index)
        assert torch.allclose(recovered, x, atol=1e-10), f"step {index}"


def test_round_trip_residual_collapses_to_b_times_prediction_gap(solver):
    """The cycle residual is B * (student - teacher), the identity the loss is built on.

    Inverting with the student's prediction and generating back with the teacher's must leave
    exactly `B * (e - v)`: the A of the generation step cancels the 1/A of the inversion, so the
    starting latent passes through untouched and only the prediction difference survives.
    """
    torch.manual_seed(0)
    x_clean = torch.randn(3, 4, 8, dtype=torch.float64)
    student = torch.randn(3, 4, 8, dtype=torch.float64)
    teacher = torch.randn(3, 4, 8, dtype=torch.float64)
    for index in range(solver.invertible_steps):
        a, b = solver.coefficients(index)
        z_hat = (x_clean - b * student) / a
        residual = x_clean - (a * z_hat + b * teacher)
        assert torch.allclose(residual, b * (student - teacher), atol=1e-10), f"step {index}"


def test_exact_student_gives_zero_round_trip(solver):
    """When the student reproduces the teacher, the round trip is exact and the loss vanishes."""
    torch.manual_seed(0)
    x_clean = torch.randn(3, 4, 8, dtype=torch.float64)
    prediction = torch.randn(3, 4, 8, dtype=torch.float64)
    for index in range(solver.invertible_steps):
        a, b = solver.coefficients(index)
        z_hat = (x_clean - b * prediction) / a
        assert torch.allclose(a * z_hat + b * prediction, x_clean, atol=1e-10), f"step {index}"


def test_multi_step_round_trip_is_exact_with_consistent_predictions(solver):
    """A k-step inversion followed by k generation steps returns the start, for any k."""
    torch.manual_seed(0)
    top = solver.invertible_steps
    x_clean = torch.randn(2, 4, 8, dtype=torch.float64)
    predictions = {i: torch.randn(2, 4, 8, dtype=torch.float64) for i in range(top)}
    for k in (1, 2, 3):
        x, indices = x_clean, []
        for step in range(k):
            index = top - 1 - step
            indices.append(index)
            x = solver.inverse(x, predictions[index], index)
        for index in reversed(indices):
            x = solver.forward(x, predictions[index], index)
        assert torch.allclose(x, x_clean, atol=1e-9), f"k={k}"
