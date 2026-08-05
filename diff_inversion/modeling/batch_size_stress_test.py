#!/usr/bin/env python3

"""Find the largest SD1.5 LoRA training micro-batch that fits on one GPU."""

from __future__ import annotations

import gc
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.train import SDXLInversionTrainer, get_lora_config
from diff_inversion.utils import make_pipe


def _make_batch(dataset: LatentTrajectoryDataset, batch_size: int) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return next(iter(loader))


def _synthetic_batch(trainer: SDXLInversionTrainer, batch_size: int) -> dict[str, torch.Tensor]:
    pipe = trainer.pipe
    latent_height = trainer.height // 8
    latent_width = trainer.width // 8
    return {
        "x_clean": torch.randn(batch_size, 4, latent_height, latent_width),
        "timestep": torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,)),
        "prompt_embeds": torch.randn(
            batch_size,
            pipe.tokenizer.model_max_length,
            pipe.text_encoder.config.hidden_size,
        ),
        "target_eps": torch.randn(batch_size, 4, latent_height, latent_width),
    }


def _try_batch(
    trainer: SDXLInversionTrainer,
    dataset: LatentTrajectoryDataset | None,
    batch_size: int,
) -> tuple[bool, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        batch = _make_batch(dataset, batch_size) if dataset is not None else _synthetic_batch(trainer, batch_size)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.forward_loss(batch).backward()
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        trainer.optimizer.zero_grad(set_to_none=True)
        del batch
        return True, peak_gib
    except torch.cuda.OutOfMemoryError:
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, 0.0
    finally:
        gc.collect()


@hydra.main(config_path="../../config", config_name="train_sd15", version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this stress test.")

    max_batch_size = int(OmegaConf.select(cfg, "stress_test.max_batch_size", default=128))
    device_name = torch.cuda.get_device_name()
    logger.info("Testing SD1.5 LoRA micro-batch capacity on {}", device_name)

    pipe = make_pipe(cfg.model, "cuda")
    trainer = SDXLInversionTrainer(
        pipe=pipe,
        lora_config=get_lora_config(cfg.lora),
        tracker=None,
        checkpoint_dir=Path(cfg.checkpoint_dir),
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_config=cfg.get("lr_scheduler"),
        gradient_accumulation_steps=1,
        num_inference_steps=cfg.model.num_inference_steps,
        height=cfg.model.height,
        width=cfg.model.width,
        save_every_steps=1,
        eval_every_steps=1,
        log_every_steps=1,
        max_val_batches=1,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    use_synthetic = bool(OmegaConf.select(cfg, "stress_test.synthetic", default=True))
    dataset = None
    if not use_synthetic:
        dataset = LatentTrajectoryDataset(
            cfg.data.root_dir,
            latents_file_name=cfg.data.latents_file_name,
            conditioning_file_name=cfg.data.conditioning_file_name,
            targets_dir_name=cfg.data.targets_dir_name,
            target_eps_file_name=cfg.data.target_eps_file_name,
            require_training_cache=cfg.data.require_training_cache,
        )
        if not dataset:
            raise RuntimeError("The training dataset is empty.")

    lower, upper = 0, 1
    while upper <= max_batch_size:
        fits, peak_gib = _try_batch(trainer, dataset, upper)
        logger.info("batch_size={} fits={} peak_allocated_gib={:.2f}", upper, fits, peak_gib)
        if not fits:
            break
        lower, upper = upper, upper * 2

    upper = min(upper, max_batch_size + 1)
    while upper - lower > 1:
        candidate = (lower + upper) // 2
        fits, peak_gib = _try_batch(trainer, dataset, candidate)
        logger.info("batch_size={} fits={} peak_allocated_gib={:.2f}", candidate, fits, peak_gib)
        if fits:
            lower = candidate
        else:
            upper = candidate

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(
        "largest_micro_batch={} (search limit={}, GPU total memory={:.2f} GiB)",
        lower,
        max_batch_size,
        total_gib,
    )


if __name__ == "__main__":
    main()
