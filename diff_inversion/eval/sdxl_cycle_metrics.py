"""Evaluate the one-transition conditional cycle for an SDXL inversion LoRA."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.eval.lora import configure_unet_lora, set_unet_lora_enabled
from diff_inversion.eval.sd15_cycle_metrics import (
    _data_roots,
    _data_source_names,
    _generation_step,
    _inverse_step,
    _per_item_metrics,
    _resolve_path,
    _summary,
)
from diff_inversion.utils import make_pipe


def _predict_eps(
    pipe,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
) -> torch.Tensor:
    model_input = pipe.scheduler.scale_model_input(latents, timesteps)
    return pipe.unet(
        model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        },
        return_dict=False,
    )[0]


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
    max_batches = None if cfg.max_batches is None else int(cfg.max_batches)
    batches = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="SDXL cycle metrics")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches += 1
        x_clean = batch["x_clean"].to(device=device, dtype=pipe.unet.dtype)
        timesteps = batch["timestep"].to(device=device, dtype=torch.long).flatten()
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(
            device=device, dtype=pipe.unet.dtype
        )
        add_time_ids = batch["add_time_ids"].to(device=device, dtype=pipe.unet.dtype)
        target_eps = batch["target_eps"].to(device=device, dtype=pipe.unet.dtype)

        student_eps = _predict_eps(
            pipe, x_clean, timesteps, prompt_embeds, pooled_prompt_embeds, add_time_ids
        )
        x_noisy = _inverse_step(pipe.scheduler, x_clean, student_eps, timesteps)
        set_unet_lora_enabled(pipe, False)
        try:
            base_eps = _predict_eps(
                pipe, x_noisy, timesteps, prompt_embeds, pooled_prompt_embeds, add_time_ids
            )
        finally:
            set_unet_lora_enabled(pipe, True)
        x_reconstructed = _generation_step(pipe.scheduler, x_noisy, base_eps, timesteps)

        metrics = {
            "inversion_mse": (student_eps.float() - target_eps.float())
            .flatten(start_dim=1)
            .square()
            .mean(dim=1),
            **{
                f"cycle_{name}": metric
                for name, metric in _per_item_metrics(x_clean, x_reconstructed).items()
            },
        }
        for name, metric in metrics.items():
            values[name].extend(metric.detach().cpu().tolist())

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
        "aggregate": {name: _summary(metric_values) for name, metric_values in sorted(values.items())},
    }


@hydra.main(config_path="../../config", config_name="eval/sdxl_cycle_metrics", version_base=None)
def main(cfg: DictConfig) -> None:
    results = evaluate_cycle_metrics(cfg)
    output_path = _resolve_path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote SDXL cycle metrics to {}", output_path)


if __name__ == "__main__":
    main()
