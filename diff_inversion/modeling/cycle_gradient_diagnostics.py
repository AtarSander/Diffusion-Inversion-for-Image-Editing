#!/usr/bin/env python3

"""Measure conditional-cycle loss and gradient scales without updating the LoRA."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import default_collate

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.train import SDXLInversionTrainer, get_lora_config
from diff_inversion.utils import make_pipe


def loss_gradient_geometry(
    loss_inversion: torch.Tensor,
    loss_cycle: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    recon_lambda: float,
) -> dict[str, float]:
    """Return raw and weighted gradient geometry for the two cycle objectives."""
    inversion_gradients = torch.autograd.grad(
        loss_inversion,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    cycle_gradients = torch.autograd.grad(
        loss_cycle,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )

    inversion_squared = torch.zeros((), device=loss_inversion.device, dtype=torch.float64)
    cycle_squared = torch.zeros_like(inversion_squared)
    dot_product = torch.zeros_like(inversion_squared)
    for inversion_gradient, cycle_gradient in zip(inversion_gradients, cycle_gradients):
        if inversion_gradient is not None:
            inversion_squared += inversion_gradient.detach().double().square().sum()
        if cycle_gradient is not None:
            cycle_squared += cycle_gradient.detach().double().square().sum()
        if inversion_gradient is not None and cycle_gradient is not None:
            dot_product += (
                inversion_gradient.detach().double() * cycle_gradient.detach().double()
            ).sum()

    inversion_norm = math.sqrt(float(inversion_squared.cpu()))
    cycle_norm = math.sqrt(float(cycle_squared.cpu()))
    weighted_cycle_norm = abs(float(recon_lambda)) * cycle_norm
    gradient_ratio = (
        weighted_cycle_norm / inversion_norm if inversion_norm > 0.0 else float("nan")
    )
    cosine = (
        float(dot_product.cpu()) / (inversion_norm * cycle_norm)
        if inversion_norm > 0.0 and cycle_norm > 0.0
        else float("nan")
    )
    combined_squared = (
        inversion_norm**2
        + weighted_cycle_norm**2
        + 2.0 * float(recon_lambda) * float(dot_product.cpu())
    )

    inversion_loss = float(loss_inversion.detach().float().cpu())
    cycle_loss = float(loss_cycle.detach().float().cpu())
    return {
        "loss_inversion": inversion_loss,
        "loss_cycle": cycle_loss,
        "weighted_loss_cycle": float(recon_lambda) * cycle_loss,
        "weighted_loss_ratio": (
            float(recon_lambda) * cycle_loss / inversion_loss
            if inversion_loss > 0.0
            else float("nan")
        ),
        "grad_norm_inversion": inversion_norm,
        "grad_norm_cycle": cycle_norm,
        "weighted_grad_norm_cycle": weighted_cycle_norm,
        "weighted_grad_norm_ratio": gradient_ratio,
        "grad_cosine": cosine,
        "grad_norm_combined": math.sqrt(max(0.0, combined_squared)),
    }


def _finite_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = [
        key
        for key, value in rows[0].items()
        if key not in {"step_idx", "inversion_step", "timestep"}
        and isinstance(value, (int, float))
    ]
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        if not values:
            continue
        summary[name] = {
            "mean": fmean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _dataset_sample_starts(dataset: LatentTrajectoryDataset) -> list[int]:
    starts = [0]
    starts.extend(dataset.cumulative_lengths[:-1])
    return starts


def _measure_state(
    trainer: SDXLInversionTrainer,
    dataset: LatentTrajectoryDataset,
    sample_positions: list[int],
    recon_lambda: float,
    label: str,
) -> dict[str, Any]:
    starts = _dataset_sample_starts(dataset)
    transition_counts = {
        int(dataset.samples[position]["num_transitions"]) for position in sample_positions
    }
    if len(transition_counts) != 1:
        raise ValueError(
            "Selected diagnostic trajectories have different transition counts: "
            f"{sorted(transition_counts)}"
        )
    transition_count = transition_counts.pop()
    rows: list[dict[str, Any]] = []

    for step_idx in range(transition_count):
        indices = [starts[position] + step_idx for position in sample_positions]
        batch = default_collate([dataset[index] for index in indices])
        _, metrics = trainer.forward_loss_with_metrics(batch)
        geometry = loss_gradient_geometry(
            metrics["loss_inversion"],
            metrics["loss_cycle"],
            trainer.trainable_parameters,
            recon_lambda,
        )
        row = {
            "step_idx": step_idx,
            "inversion_step": transition_count - 1 - step_idx,
            "timestep": int(batch["timestep"][0]),
            **geometry,
        }
        rows.append(row)
        logger.info(
            "{} step={:02d} timestep={} loss_inv={:.6g} loss_cycle={:.6g} "
            "grad_inv={:.6g} grad_cycle={:.6g} weighted_grad_ratio={:.6g} cosine={:.6g}",
            label,
            row["inversion_step"],
            row["timestep"],
            row["loss_inversion"],
            row["loss_cycle"],
            row["grad_norm_inversion"],
            row["grad_norm_cycle"],
            row["weighted_grad_norm_ratio"],
            row["grad_cosine"],
        )
        del batch, metrics

    return {"summary": _finite_summary(rows), "per_timestep": rows}


@hydra.main(config_path="../../config", config_name="train_sd15", version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cycle gradient diagnostics.")
    if str(cfg.training_target.mode) != "conditional_cycle":
        raise ValueError("Cycle gradient diagnostics require training_target=conditional_cycle.")

    recon_lambda = float(cfg.training_target.recon_lambda)
    num_trajectories = int(
        OmegaConf.select(cfg, "diagnostics.num_trajectories", default=4)
    )
    seed = int(cfg.get("seed", 42))
    output_path = Path(str(OmegaConf.select(cfg, "diagnostics.output_path")))
    trained_checkpoint = Path(str(OmegaConf.select(cfg, "diagnostics.trained_checkpoint"))).resolve(strict=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pipe = make_pipe(cfg.model, "cuda")
    trainer = SDXLInversionTrainer(
        pipe=pipe,
        lora_config=get_lora_config(cfg.lora),
        tracker=None,
        checkpoint_dir=cfg.checkpoint_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_config=cfg.get("lr_scheduler"),
        gradient_accumulation_steps=1,
        num_inference_steps=cfg.model.num_inference_steps,
        height=cfg.model.height,
        width=cfg.model.width,
        save_every_steps=0,
        eval_every_steps=0,
        log_every_steps=0,
        max_val_batches=0,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        offload_frozen_components=cfg.offload_frozen_components,
        training_target_mode=str(cfg.training_target.mode),
        training_guidance_scale=cfg.training_target.guidance_scale,
        recon_lambda=recon_lambda,
    )
    dataset = LatentTrajectoryDataset(
        cfg.data.root_dir,
        latents_file_name=cfg.data.latents_file_name,
        conditioning_file_name=cfg.data.conditioning_file_name,
        targets_dir_name=cfg.data.targets_dir_name,
        target_eps_file_name=cfg.data.target_eps_file_name,
        require_training_cache=cfg.data.require_training_cache,
    )
    if len(dataset.samples) < num_trajectories:
        raise ValueError(
            f"Requested {num_trajectories} trajectories from only {len(dataset.samples)} samples."
        )

    generator = torch.Generator().manual_seed(seed)
    sample_positions = torch.randperm(len(dataset.samples), generator=generator)[
        :num_trajectories
    ].tolist()
    selected_samples = [
        {
            "sample_position": int(position),
            "sample_idx": dataset.samples[position]["sample_idx"],
            "sample_dir": str(dataset.samples[position]["sample_dir"]),
        }
        for position in sample_positions
    ]

    logger.info("Measuring initialized r16 adapter on {} trajectories", num_trajectories)
    initialized = _measure_state(
        trainer,
        dataset,
        sample_positions,
        recon_lambda,
        "initialized",
    )
    trainer.load_checkpoint(trained_checkpoint)
    logger.info("Measuring trained adapter from {}", trained_checkpoint)
    trained = _measure_state(
        trainer,
        dataset,
        sample_positions,
        recon_lambda,
        "trained_lambda1",
    )

    result = {
        "seed": seed,
        "rank": int(cfg.lora.r),
        "recon_lambda": recon_lambda,
        "num_trajectories": num_trajectories,
        "num_transition_positions": len(initialized["per_timestep"]),
        "trained_checkpoint": str(trained_checkpoint),
        "selected_samples": selected_samples,
        "initialized": initialized,
        "trained_lambda1": trained,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Cycle gradient diagnostics saved to {}", output_path)
    logger.info("Initialized summary: {}", initialized["summary"])
    logger.info("Trained lambda-1 summary: {}", trained["summary"])


if __name__ == "__main__":
    main()
