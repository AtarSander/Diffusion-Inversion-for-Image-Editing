from types import MethodType, SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.one_step_endpoint import endpoint_latent_from_noise, endpoint_noise_target
from diff_inversion.modeling.train import SDXLInversionTrainer, build_train_sampler


def _scheduler() -> DDIMScheduler:
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(50)
    return scheduler


def _write_endpoint_sample(root, *, include_target_eps=False):
    sample = root / "sample_000000"
    (sample / "latents").mkdir(parents=True)
    (sample / "targets").mkdir()
    trajectory = torch.stack([torch.full((1, 4, 2, 2), 9.0), torch.ones((1, 4, 2, 2)), torch.full((1, 4, 2, 2), 3.0)])
    torch.save(trajectory, sample / "latents" / "trajectory.pt")
    torch.save({"prompt_embeds": torch.ones(1, 2, 3)}, sample / "conditioning.pt")
    if include_target_eps:
        torch.save(torch.zeros(2, 4, 2, 2), sample / "targets" / "target_eps.pt")
    (sample / "timesteps.json").write_text("[981, 981, 0]\n")
    (sample / "meta.json").write_text('{"trajectory_length": 3, "sample_idx": 0}\n')


def test_endpoint_dataset_returns_one_z0_to_z50_pair_without_target_eps(tmp_path):
    _write_endpoint_sample(tmp_path)
    dataset = LatentTrajectoryDataset(
        tmp_path, sample_mode="endpoint", require_training_cache=True, require_target_eps=False
    )
    assert len(dataset) == 1
    item = dataset[0]
    torch.testing.assert_close(item["x_clean"], torch.full((4, 2, 2), 3.0))
    torch.testing.assert_close(item["target_endpoint"], torch.full((4, 2, 2), 9.0))
    assert item["timestep"].item() == 0
    assert item["terminal_timestep"].item() == 981
    assert dataset.transition_metadata(0)["sample_mode"] == "endpoint"


def test_transition_dataset_cardinality_is_unchanged(tmp_path):
    _write_endpoint_sample(tmp_path, include_target_eps=True)
    dataset = LatentTrajectoryDataset(tmp_path, require_training_cache=True)
    assert len(dataset) == 2
    assert dataset[1]["step_idx"] == 1


def test_endpoint_projection_and_noise_target_are_exact_inverse():
    scheduler = _scheduler()
    z0 = torch.randn(2, 4, 3, 3)
    eps = torch.randn_like(z0)
    terminal = scheduler.timesteps[0].expand(2)
    endpoint = endpoint_latent_from_noise(z0, eps, terminal, scheduler)
    recovered = endpoint_noise_target(z0, endpoint, terminal, scheduler)
    torch.testing.assert_close(recovered, eps, rtol=1e-5, atol=1e-6)


def test_endpoint_loss_is_direct_terminal_loss_and_backpropagates():
    scheduler = _scheduler()
    trainer = object.__new__(SDXLInversionTrainer)
    trainer.pipe = SimpleNamespace(scheduler=scheduler)
    trainer.scale = torch.nn.Parameter(torch.tensor(0.2))

    def predict_noise(self, latents, *args):
        return torch.ones_like(latents) * self.scale

    trainer.predict_noise = MethodType(predict_noise, trainer)
    z0 = torch.randn(2, 4, 3, 3)
    terminal = scheduler.timesteps[0].expand(2)
    target = endpoint_latent_from_noise(z0, torch.full_like(z0, 0.4), terminal, scheduler)
    loss, metrics = trainer._forward_endpoint_loss(
        x_clean=z0, input_timesteps=scheduler.timesteps[-1].expand(2),
        terminal_timesteps=terminal, prompt_embeds=torch.empty(2, 1, 1),
        pooled_prompt_embeds=None, add_time_ids=None, target_endpoint=target,
    )
    assert set(metrics) == {"loss", "loss_endpoint", "loss_endpoint_eps", "terminal_timestep"}
    assert loss > 0
    loss.backward()
    assert trainer.scale.grad is not None and torch.isfinite(trainer.scale.grad)


def test_endpoint_rejects_mismatched_schedule():
    scheduler = _scheduler()
    trainer = object.__new__(SDXLInversionTrainer)
    trainer.pipe = SimpleNamespace(scheduler=scheduler)
    trainer.predict_noise = MethodType(lambda self, latents, *args: torch.zeros_like(latents), trainer)
    with pytest.raises(ValueError, match="terminal timestep"):
        trainer._forward_endpoint_loss(
            x_clean=torch.randn(1, 4, 2, 2),
            input_timesteps=scheduler.timesteps[-1:].clone(),
            terminal_timesteps=torch.tensor([999]), prompt_embeds=torch.empty(1, 1, 1),
            pooled_prompt_embeds=None, add_time_ids=None, target_endpoint=torch.randn(1, 4, 2, 2),
        )


def test_endpoint_rejects_final_tail_sampling():
    dataset = object.__new__(LatentTrajectoryDataset)
    dataset.sample_mode = "endpoint"
    with pytest.raises(ValueError, match="uniform only"):
        build_train_sampler(dataset, {"mode": "final_tail"})
