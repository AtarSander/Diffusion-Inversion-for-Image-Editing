from types import MethodType, SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler

from diff_inversion.modeling.train import SDXLInversionTrainer


class _FakeCheckpointedModel:
    def __init__(self) -> None:
        self.is_gradient_checkpointing = True
        self.checkpoint_calls: list[str] = []

    def disable_gradient_checkpointing(self) -> None:
        self.checkpoint_calls.append("disable")
        self.is_gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.checkpoint_calls.append("enable")
        self.is_gradient_checkpointing = True


def _trainer_with_scheduler() -> SDXLInversionTrainer:
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
    trainer = object.__new__(SDXLInversionTrainer)
    trainer.pipe = SimpleNamespace(scheduler=scheduler)
    return trainer


def test_ddim_generation_step_matches_diffusers_for_mixed_timesteps() -> None:
    trainer = _trainer_with_scheduler()
    scheduler = trainer.pipe.scheduler
    timesteps = scheduler.timesteps[torch.tensor([0, 17, 49])]
    x_noisy = torch.randn(3, 4, 8, 8)
    eps = torch.randn_like(x_noisy)

    actual = trainer._ddim_generation_step(x_noisy, eps, timesteps)
    expected = torch.cat(
        [
            scheduler.step(
                model_output=eps[index : index + 1],
                timestep=int(timestep.item()),
                sample=x_noisy[index : index + 1],
            ).prev_sample
            for index, timestep in enumerate(timesteps)
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_ddim_inverse_and_generation_are_an_exact_transition_pair() -> None:
    trainer = _trainer_with_scheduler()
    timesteps = trainer.pipe.scheduler.timesteps[torch.tensor([0, 17, 49])]
    x_clean = torch.randn(3, 4, 8, 8)
    eps = torch.randn_like(x_clean, requires_grad=True)

    x_noisy = trainer._ddim_inverse_step(x_clean, eps, timesteps)
    reconstructed = trainer._ddim_generation_step(x_noisy, eps, timesteps)

    torch.testing.assert_close(reconstructed, x_clean, rtol=1e-5, atol=2e-6)
    x_noisy.square().mean().backward()
    assert eps.grad is not None
    assert torch.isfinite(eps.grad).all()
    assert eps.grad.abs().sum() > 0


def test_cycle_rejects_timesteps_outside_the_active_schedule() -> None:
    trainer = _trainer_with_scheduler()

    with pytest.raises(ValueError, match="absent from the DDIM schedule"):
        trainer._ddim_inverse_step(
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
            torch.tensor([999]),
        )


def test_conditional_cycle_disables_lora_only_for_base_generation_and_backpropagates() -> None:
    trainer = _trainer_with_scheduler()
    trainer.model = _FakeCheckpointedModel()
    trainer.recon_lambda = 0.5
    trainer.adapters_enabled = True
    trainer.adapter_calls = []
    trainer.base_inputs = []
    trainer.student_scale = torch.nn.Parameter(torch.tensor(0.3))

    def set_lora_enabled(self: SDXLInversionTrainer, enabled: bool) -> None:
        self.adapter_calls.append(enabled)
        self.adapters_enabled = enabled

    def predict_noise(
        self: SDXLInversionTrainer,
        latents: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        del scheduler_timesteps, prompt_embeds, pooled_prompt_embeds, add_time_ids
        if self.adapters_enabled:
            return torch.ones_like(latents) * self.student_scale
        self.base_inputs.append(latents)
        return 0.2 * latents + 0.1

    trainer._set_lora_enabled = MethodType(set_lora_enabled, trainer)
    trainer.predict_noise = MethodType(predict_noise, trainer)

    x_clean = torch.randn(2, 4, 8, 8)
    timesteps = trainer.pipe.scheduler.timesteps[torch.tensor([4, 31])]
    target_eps = torch.full_like(x_clean, 0.3)
    expected_noisy = trainer._ddim_inverse_step(
        x_clean,
        torch.full_like(x_clean, 0.3),
        timesteps,
    )

    loss, metrics = trainer._forward_loss_regularized(
        x_clean=x_clean,
        scheduler_timesteps=timesteps,
        prompt_embeds=torch.empty(2, 1, 1),
        pooled_prompt_embeds=None,
        add_time_ids=None,
        target_eps=target_eps,
    )

    assert trainer.adapter_calls == [False, True]
    assert trainer.model.checkpoint_calls == ["disable", "enable"]
    assert trainer.model.is_gradient_checkpointing
    assert trainer.adapters_enabled
    assert len(trainer.base_inputs) == 1
    torch.testing.assert_close(trainer.base_inputs[0], expected_noisy)
    assert set(metrics) == {"loss", "loss_inversion", "loss_cycle"}
    torch.testing.assert_close(
        metrics["loss"],
        metrics["loss_inversion"] + trainer.recon_lambda * metrics["loss_cycle"],
    )
    torch.testing.assert_close(metrics["loss_inversion"], torch.zeros(()))

    loss.backward()
    assert trainer.student_scale.grad is not None
    assert torch.isfinite(trainer.student_scale.grad)
    assert trainer.student_scale.grad.abs() > 0


def test_conditional_cycle_is_a_supported_training_target_mode() -> None:
    assert SDXLInversionTrainer._normalize_training_target_mode("conditional_cycle") == (
        "conditional_cycle"
    )
