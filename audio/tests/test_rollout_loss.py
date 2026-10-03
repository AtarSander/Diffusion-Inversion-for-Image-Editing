# ABOUTME: Tests the rollout loss on an exact toy flow: zero for the exact student, its first step
# ABOUTME: equal to the one-step loss, errors compounding over steps, and checkpointing exact.

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.stable_audio import ExactDPMSolver  # noqa: E402
from src.inversion_lora.train_stable_audio import StableAudioInversionTrainer  # noqa: E402


class FakeCosineScheduler:
    """A geometric sigma grid with the EDM alpha convention, the shape Stable Audio's has."""

    def __init__(self, num_steps: int = 10, sigma_max: float = 100.0, sigma_min: float = 0.3):
        ratio = (sigma_min / sigma_max) ** (1.0 / (num_steps - 1))
        sigmas = torch.tensor([sigma_max * ratio**i for i in range(num_steps)], dtype=torch.float64)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])
        self.timesteps = (sigmas.atan() * 2.0 / torch.pi).tolist()

    def _sigma_to_alpha_sigma_t(self, sigma):
        return torch.ones_like(sigma), sigma

K, J, SHAPE = 3, 6, (2, 4, 8)  # rollout steps, grid index of the input, [B, C, L]


def toy_chain(solver):
    """A linear teacher D(x, i) = c_i x and its exact reverse trajectory."""
    n = len(solver.timesteps)
    c = torch.linspace(0.3, 0.9, n, dtype=torch.float64)
    x = [torch.randn(SHAPE, generator=torch.Generator().manual_seed(0), dtype=torch.float64)]
    for i in range(n - 1):
        x.append(solver.forward(x[-1], c[i] * x[-1], i))
    return c, x


def fake_trainer(solver, c, scale, checkpoint):
    """Just what rollout_per_example touches; the student is the exact inverse times `scale`."""

    def student(x, index, text_audio, timing):
        i = int(index[0])  # the cleaner state's grid index; exact answer is D(x_{i-1}, i-1)
        a, b = solver.coefficients(i - 1)
        return scale * c[i - 1] / (a + b * c[i - 1]) * x

    return types.SimpleNamespace(
        device=torch.device("cpu"), unet=types.SimpleNamespace(dtype=torch.float64),
        solver=solver, rollout_steps=K, rollout_checkpoint=checkpoint,
        timing_states=lambda batch: None, data_prediction_at=student,
    ), student


def rollout(scale, checkpoint=False):
    solver = ExactDPMSolver(FakeCosineScheduler(num_steps=12))
    c, x = toy_chain(solver)
    trainer, student = fake_trainer(solver, c, scale, checkpoint)
    index = torch.full((SHAPE[0],), J)
    batch = {
        "x_clean": x[J],
        "timestep": torch.tensor([solver.timesteps[J]] * SHAPE[0]),
        "text_audio": torch.zeros(SHAPE[0], 1, 1),
        "x_noisier": torch.stack([x[J - m] for m in range(1, K + 1)], dim=1),
    }
    first = student(x[J], index, None, None)
    terms = StableAudioInversionTrainer.rollout_per_example(trainer, batch, first)
    one_step = ((first - c[J - 1] * x[J - 1]) ** 2).flatten(1).mean(dim=1)
    return terms, one_step


def test_exact_student_has_zero_rollout_loss():
    terms, _ = rollout(scale=1.0)
    assert terms.shape == (SHAPE[0], K)
    assert terms.abs().max() < 1e-20


def test_first_step_is_the_one_step_loss_and_errors_compound():
    terms, one_step = rollout(scale=1.05)
    assert torch.allclose(terms[:, 0].double(), one_step, rtol=1e-5)  # terms are float32
    assert (terms[:, 1:].mean(0) > 0).all() and terms[:, -1].mean() > terms[:, 0].mean()


def test_gradient_reaches_every_step_and_checkpointing_is_exact():
    losses, grads = [], []
    for checkpoint in (False, True):
        scale = torch.tensor(1.05, dtype=torch.float64, requires_grad=True)
        terms, _ = rollout(scale, checkpoint)
        loss = terms[:, -1].mean()  # the last state depends on every step's prediction
        (grad,) = torch.autograd.grad(loss, scale)
        losses.append(loss.detach())
        grads.append(grad)
    assert grads[0].abs() > 0
    assert torch.allclose(losses[0], losses[1]) and torch.allclose(grads[0], grads[1])
