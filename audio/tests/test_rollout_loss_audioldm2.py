# ABOUTME: Tests AudioLDM2's DDIM rollout loss on an exact toy chain: zero for the exact student, its
# ABOUTME: first step equal to the one-step loss, errors compounding, checkpointing exact, both losses.

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.train import AudioLDM2InversionTrainer  # noqa: E402

T, N = 1000, 10  # num_train_timesteps, grid length
STRIDE = T // N
K, I, SHAPE = 3, 6, (2, 4, 6, 5)  # rollout steps, transition index, [B, C, H, W]
TIMESTEPS = [1 + STRIDE * (N - 1 - j) for j in range(N)]  # leading spacing, offset 1: 901..1


def toy_scheduler():
    betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    return types.SimpleNamespace(alphas_cumprod=alphas, final_alpha_cumprod=alphas[0],
                                 scale_model_input=lambda x, t: x)


def alpha_pair(scheduler, t):
    prev = t - STRIDE
    a_prev = scheduler.alphas_cumprod[prev] if prev >= 0 else scheduler.final_alpha_cumprod
    return scheduler.alphas_cumprod[t], a_prev


def toy_chain(scheduler):
    """A linear teacher eps(x, t) = c_t x and its exact reverse DDIM trajectory, noisy first."""
    c = {t: 0.2 + 0.6 * j / N for j, t in enumerate(TIMESTEPS)}
    x = [torch.randn(SHAPE, generator=torch.Generator().manual_seed(0), dtype=torch.float64)]
    for t in TIMESTEPS:
        a_t, a_prev = alpha_pair(scheduler, t)
        eps = c[t] * x[-1]
        x0 = (x[-1] - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x.append(a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps)
    return c, x


def fake_trainer(scheduler, c, scale, checkpoint, weight=1.0):
    """What training_loss and rollout touch; the student is the exact shifted epsilon x `scale`.

    The exact answer at the cleaner x_prev for timestep t solves eps = c_t (A x_prev + B eps).
    """
    trainer = types.SimpleNamespace(
        device=torch.device("cpu"), unet=types.SimpleNamespace(dtype=torch.float64),
        num_train_timesteps=T, cfg=types.SimpleNamespace(num_inference_steps=N),
        ldm=types.SimpleNamespace(model=types.SimpleNamespace(scheduler=scheduler)),
        rollout_steps=K, rollout_checkpoint=checkpoint, rollout_weight=weight,
    )
    trainer.ddim_inversion_coefficients = types.MethodType(
        AudioLDM2InversionTrainer.ddim_inversion_coefficients, trainer)
    trainer.rollout_per_example = types.MethodType(
        AudioLDM2InversionTrainer.rollout_per_example, trainer)

    def student(x, timestep, batch):
        a, b = trainer.ddim_inversion_coefficients(timestep, x.dim())
        ct = torch.tensor([c[int(t)] for t in timestep], dtype=torch.float64).reshape(a.shape)
        return scale * ct * a / (1 - ct * b) * x

    trainer.conditional_eps = student
    trainer.predict_noise = lambda batch: student(batch["x_clean"], batch["timestep"], batch)
    trainer.target_noise = lambda batch: batch["target_eps"]
    return trainer


def make_batch(c, x):
    # Transition I: the student sees x[I + 1] at TIMESTEPS[I]; the stored state above is x[I].
    t = TIMESTEPS[I]
    return {
        "x_clean": x[I + 1],
        "timestep": torch.full((SHAPE[0],), t, dtype=torch.long),
        "target_eps": c[t] * x[I],
        "x_noisier": torch.stack([x[I + 1 - m] for m in range(1, K + 1)], dim=1),
    }


def rollout(scale, checkpoint=False):
    scheduler = toy_scheduler()
    c, x = toy_chain(scheduler)
    trainer = fake_trainer(scheduler, c, scale, checkpoint)
    batch = make_batch(c, x)
    first = trainer.predict_noise(batch)
    terms = trainer.rollout_per_example(batch, first)
    one_step = ((first - batch["target_eps"]) ** 2).flatten(1).mean(dim=1)
    return terms, one_step


def test_exact_student_has_zero_rollout_loss():
    terms, one_step = rollout(scale=1.0)
    assert terms.shape == (SHAPE[0], K)
    assert one_step.abs().max() < 1e-20
    assert terms.abs().max() < 1e-12  # float32 terms of float64 states


def test_first_step_is_the_one_step_loss_and_errors_compound():
    terms, one_step = rollout(scale=1.05)
    assert torch.allclose(terms[:, 0].double(), one_step, rtol=1e-5)
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


def test_training_loss_adds_the_weighted_rollout_to_the_one_step_loss():
    scheduler = toy_scheduler()
    c, x = toy_chain(scheduler)
    trainer = fake_trainer(scheduler, c, scale=1.05, checkpoint=False, weight=0.5)
    batch = make_batch(c, x)
    loss, per_example, extra = AudioLDM2InversionTrainer.training_loss(trainer, batch)
    terms = trainer.rollout_per_example(batch, trainer.predict_noise(batch))
    assert torch.allclose(per_example.mean(), torch.tensor(extra["train/loss_inv"],
                                                           dtype=per_example.dtype))
    assert torch.allclose(loss.double(), per_example.mean().double() + 0.5 * terms.mean().double())
    assert set(extra) == {"train/loss_inv", "train/loss_rollout",
                          *(f"train/rollout_m{m}" for m in range(1, K + 1))}


def test_disabled_rollout_is_the_plain_one_step_loss():
    scheduler = toy_scheduler()
    c, x = toy_chain(scheduler)
    trainer = fake_trainer(scheduler, c, scale=1.05, checkpoint=False)
    trainer.rollout_steps = 0
    trainer.per_example_loss = lambda batch: (
        (trainer.predict_noise(batch) - batch["target_eps"]) ** 2).flatten(1).mean(dim=1)
    batch = make_batch(c, x)
    loss, per_example, extra = AudioLDM2InversionTrainer.training_loss(trainer, batch)
    assert extra == {} and torch.equal(loss, per_example.mean())
