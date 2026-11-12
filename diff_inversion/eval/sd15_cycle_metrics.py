"""Evaluate the one-transition latent cycle used by conditional-cycle training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.eval.lora import configure_unet_lora, set_unet_lora_enabled
from diff_inversion.utils import make_pipe


def _resolve_path(value: str | Path) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _data_roots(cfg: DictConfig) -> str | list[str]:
    roots = OmegaConf.select(cfg, "data.val_root_dirs", default=None)
    if roots is not None:
        resolved = [str(root) for root in roots]
        if resolved:
            return resolved
    root = OmegaConf.select(cfg, "data.val_root_dir", default=None)
    if root is None:
        raise ValueError("Set data.val_root_dir or data.val_root_dirs for cycle evaluation.")
    return str(root)


def _data_source_names(cfg: DictConfig) -> list[str] | None:
    names = OmegaConf.select(cfg, "data.val_source_names", default=None)
    return None if names is None else [str(name) for name in names]


def _alpha_pair(
    scheduler, timesteps: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DDIM alpha values at each noisy timestep and adjacent cleaner timestep."""
    if str(scheduler.config.prediction_type) != "epsilon":
        raise ValueError("Cycle metrics require scheduler prediction_type='epsilon'.")
    if bool(getattr(scheduler.config, "thresholding", False)) or bool(
        getattr(scheduler.config, "clip_sample", False)
    ):
        raise ValueError("Cycle metrics require DDIM thresholding=false and clip_sample=false.")

    timesteps = timesteps.flatten().to(device=reference.device, dtype=torch.long)
    schedule = scheduler.timesteps.to(device=reference.device, dtype=torch.long)
    matches = timesteps[:, None].eq(schedule[None, :])
    present = matches.any(dim=1)
    if not bool(present.all().item()):
        missing = timesteps[~present].detach().cpu().tolist()
        raise ValueError(
            f"Cycle timesteps are absent from the configured DDIM schedule: {missing}"
        )

    positions = matches.to(torch.int64).argmax(dim=1)
    cleaner_positions = positions + 1
    has_cleaner = cleaner_positions < schedule.numel()
    cleaner_timesteps = schedule[cleaner_positions.clamp(max=schedule.numel() - 1)]
    alphas = scheduler.alphas_cumprod.to(device=reference.device, dtype=reference.dtype)
    alpha_noisy = alphas[timesteps]
    alpha_clean = alphas[cleaner_timesteps]
    final_alpha = torch.as_tensor(
        scheduler.final_alpha_cumprod,
        device=reference.device,
        dtype=reference.dtype,
    )
    alpha_clean = torch.where(has_cleaner, alpha_clean, final_alpha.expand_as(alpha_clean))
    shape = (-1,) + (1,) * (reference.ndim - 1)
    return alpha_noisy.reshape(shape), alpha_clean.reshape(shape)


def _inverse_step(
    scheduler, x_clean: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    alpha_noisy, alpha_clean = _alpha_pair(scheduler, timesteps, x_clean)
    predicted_x0 = (x_clean - (1.0 - alpha_clean).sqrt() * eps) / alpha_clean.sqrt()
    return alpha_noisy.sqrt() * predicted_x0 + (1.0 - alpha_noisy).sqrt() * eps


def _generation_step(
    scheduler, x_noisy: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    alpha_noisy, alpha_clean = _alpha_pair(scheduler, timesteps, x_noisy)
    predicted_x0 = (x_noisy - (1.0 - alpha_noisy).sqrt() * eps) / alpha_noisy.sqrt()
    return alpha_clean.sqrt() * predicted_x0 + (1.0 - alpha_clean).sqrt() * eps


def _predict_eps(
    pipe, latents: torch.Tensor, timesteps: torch.Tensor, prompt_embeds: torch.Tensor
) -> torch.Tensor:
    model_input = pipe.scheduler.scale_model_input(latents, timesteps)
    return pipe.unet(
        model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]


def _per_item_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, torch.Tensor]:
    delta = (candidate - reference).float().flatten(start_dim=1)
    mse = delta.square().mean(dim=1)
    rmse = mse.sqrt()
    reference_std = (
        reference.float().flatten(start_dim=1).std(dim=1, unbiased=False).clamp_min(1e-12)
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": delta.abs().mean(dim=1),
        "relative_rmse_to_clean_std": rmse / reference_std,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


@torch.no_grad()
def evaluate_cycle_metrics(cfg: DictConfig) -> dict[str, Any]:
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path is required.")
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but CUDA is unavailable.")

    pipe = make_pipe(cfg.model, str(device))
    if not configure_unet_lora(pipe, cfg.lora):
        raise ValueError("lora.enabled=true is required for cycle evaluation.")
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(int(cfg.model.num_inference_steps), device=device)

    dataset = LatentTrajectoryDataset(
        _data_roots(cfg),
        latents_file_name=str(cfg.data.latents_file_name),
        conditioning_file_name=str(cfg.data.conditioning_file_name),
        targets_dir_name=str(cfg.data.targets_dir_name),
        target_eps_file_name=str(cfg.data.target_eps_file_name),
        target_uncond_eps_file_name=str(cfg.data.target_uncond_eps_file_name),
        require_training_cache=bool(cfg.data.require_training_cache),
        source_names=_data_source_names(cfg),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )

    values: dict[str, list[float]] = defaultdict(list)
    by_timestep: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    max_batches = cfg.max_batches
    max_batches = None if max_batches is None else int(max_batches)
    batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Cycle metrics")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches += 1
        x_clean = batch["x_clean"].to(device=device, dtype=pipe.unet.dtype)
        timesteps = batch["timestep"].to(device=device, dtype=torch.long).flatten()
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
        target_eps = batch["target_eps"].to(device=device, dtype=pipe.unet.dtype)

        student_eps = _predict_eps(pipe, x_clean, timesteps, prompt_embeds)
        x_noisy = _inverse_step(pipe.scheduler, x_clean, student_eps, timesteps)
        set_unet_lora_enabled(pipe, False)
        try:
            base_eps = _predict_eps(pipe, x_noisy, timesteps, prompt_embeds)
        finally:
            set_unet_lora_enabled(pipe, True)
        x_reconstructed = _generation_step(pipe.scheduler, x_noisy, base_eps, timesteps)

        batch_metrics = {
            "inversion_mse": (student_eps.float() - target_eps.float())
            .flatten(start_dim=1)
            .square()
            .mean(dim=1),
            **{
                f"cycle_{name}": metric
                for name, metric in _per_item_metrics(x_clean, x_reconstructed).items()
            },
        }
        for item_idx, timestep in enumerate(timesteps.detach().cpu().tolist()):
            for name, metric in batch_metrics.items():
                value = float(metric[item_idx].detach().cpu())
                values[name].append(value)
                by_timestep[int(timestep)][name].append(value)

    if not values:
        raise RuntimeError("No validation batches were evaluated.")
    return {
        "definition": {
            "inversion_mse": "MSE between conditional LoRA epsilon prediction and cached target epsilon.",
            "cycle": "LoRA inverse DDIM transition followed by frozen-base DDIM generation transition, compared with the input clean latent.",
            "precision": str(cfg.model.torch_dtype),
            "num_inference_steps": int(cfg.model.num_inference_steps),
        },
        "checkpoint_path": str(_resolve_path(cfg.checkpoint_path)),
        "num_batches": batches,
        "num_transitions": len(next(iter(values.values()))),
        "aggregate": {
            name: _summary(metric_values) for name, metric_values in sorted(values.items())
        },
        "by_timestep": {
            str(timestep): {
                name: _summary(metric_values) for name, metric_values in sorted(metrics.items())
            }
            for timestep, metrics in sorted(by_timestep.items())
        },
    }


@hydra.main(config_path="../../config", config_name="eval/sd15_cycle_metrics", version_base=None)
def main(cfg: DictConfig) -> None:
    results = evaluate_cycle_metrics(cfg)
    output_path = _resolve_path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote cycle metrics to {}", output_path)
    logger.info("Cycle aggregate: {}", results["aggregate"])


if __name__ == "__main__":
    main()
