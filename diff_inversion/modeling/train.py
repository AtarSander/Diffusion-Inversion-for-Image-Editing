"""Train a UNet LoRA adapter on saved SDXL latent trajectories."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from diffusers import StableDiffusionXLPipeline
import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    encode_prompt_sdxl,
    make_pipe,
)
from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.validation_preview import (
    log_validation_preview,
    should_run_validation_preview,
)


def _cfg_to_container(cfg: Any) -> Any:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _lora_kwargs(lora_cfg: DictConfig) -> dict[str, Any]:
    raw = _cfg_to_container(lora_cfg)
    ignored = {"checkpoint_path", "adapter_name"}
    return {key: value for key, value in raw.items() if key not in ignored and value is not None}


def _init_wandb(cfg: DictConfig):
    wandb_cfg = cfg.wandb
    if wandb_cfg.mode == "disabled":
        return None

    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning("W&B is not installed; training will continue without W&B logging")
        return None

    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        name=wandb_cfg.run_name or cfg.run_name,
        tags=_cfg_to_container(wandb_cfg.tags),
        mode=wandb_cfg.mode,
        config=_cfg_to_container(cfg),
    )
    return run


class SDXLInversionLoraTrainer:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
        cfg: DictConfig,
        tracker: Any | None,
    ) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self.tracker = tracker
        self.device = pipe.device
        self.unet = pipe.unet
        self.adapter_name = str(cfg.lora.adapter_name)
        self.checkpoint_dir = _resolve_path(cfg.checkpoint_dir)
        self.global_step = 0
        self.skipped_nonfinite_steps = 0

        self._prepare_models()
        self._prepare_lora()
        self.trainable_params = [param for param in self.unet.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )

    def _prepare_models(self) -> None:
        self.unet.requires_grad_(False)
        self.unet.train()

        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.requires_grad_(False)
            self.pipe.text_encoder.eval()
        if self.pipe.text_encoder_2 is not None:
            self.pipe.text_encoder_2.requires_grad_(False)
            self.pipe.text_encoder_2.eval()
        if self.pipe.vae is not None:
            self.pipe.vae.requires_grad_(False)
            self.pipe.vae.eval()

        if bool(self.cfg.training.gradient_checkpointing):
            self.unet.enable_gradient_checkpointing()

    def _prepare_lora(self) -> None:
        logger.info("Adding UNet LoRA adapter '{}'", self.adapter_name)
        lora_config = LoraConfig(**_lora_kwargs(self.cfg.lora))
        self.unet.add_adapter(lora_config, adapter_name=self.adapter_name)

        checkpoint_path = self.cfg.lora.checkpoint_path
        if checkpoint_path:
            checkpoint = _resolve_path(checkpoint_path)
            logger.info("Loading LoRA checkpoint from {}", checkpoint)
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
            set_peft_model_state_dict(self.unet, state_dict, adapter_name=self.adapter_name)

        if hasattr(self.unet, "enable_adapters"):
            self.unet.enable_adapters()

        if bool(self.cfg.training.cast_trainable_params_to_float32):
            self._cast_trainable_params_to_float32()

        trainable = sum(param.numel() for param in self.unet.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.unet.parameters())
        logger.info(
            "Trainable UNet parameters: {:,} / {:,} ({:.4f}%)",
            trainable,
            total,
            100 * trainable / total,
        )

    def _cast_trainable_params_to_float32(self) -> None:
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        logger.info("Casted trainable UNet parameters to float32")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        epochs = int(self.cfg.training.epochs)
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            self._log(train_metrics)

            if val_loader is not None:
                val_metrics = self.validation_epoch(val_loader, epoch)
                self._log(val_metrics)
                if should_run_validation_preview(self.cfg, epoch):
                    log_validation_preview(
                        pipe=self.pipe,
                        cfg=self.cfg,
                        tracker=self.tracker,
                        checkpoint_dir=self.checkpoint_dir,
                        epoch=epoch,
                        global_step=self.global_step,
                    )

            save_epoch_frequency = int(self.cfg.training.save_every_epochs)
            if save_epoch_frequency > 0 and (epoch + 1) % save_epoch_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        self.save_checkpoint("checkpoint_final.pt")
        if self.tracker is not None:
            self.tracker.finish()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        self.unet.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}", leave=False)

        for batch in progress:
            loss, step_metrics = self.training_step(
                batch,
                collect_metrics=bool(
                    self.cfg.logging.timestep_stats or self.cfg.logging.prediction_stats
                ),
                metrics_prefix="train",
            )
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                self.optimizer.zero_grad(set_to_none=True)
                self._handle_nonfinite_batch(
                    reason="loss",
                    value=loss_value,
                    step_metrics=step_metrics,
                )
                continue

            loss.backward()

            max_grad_norm = float(self.cfg.training.max_grad_norm)
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                grad_norm_value = float(grad_norm.detach().cpu())
            else:
                grad_norm_value = self._grad_norm()

            if not math.isfinite(grad_norm_value):
                self._handle_nonfinite_batch(
                    reason="grad_norm",
                    value=grad_norm_value,
                    step_metrics=step_metrics,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(loss_value)
            self.global_step += 1
            progress.set_postfix(loss=f"{loss_value:.6f}")

            log_frequency = int(self.cfg.logging.log_every_steps)
            if log_frequency > 0 and self.global_step % log_frequency == 0:
                log_metrics = self._build_step_log(loss_value, grad_norm_value, step_metrics)
                if log_metrics:
                    self._log(log_metrics)
            save_step_frequency = int(self.cfg.training.save_every_steps)
            if save_step_frequency > 0 and self.global_step % save_step_frequency == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        return {
            "train/loss": sum(losses) / len(losses) if losses else float("nan"),
            "epoch": float(epoch + 1),
            "global_step": float(self.global_step),
            "train/skipped_nonfinite_steps": float(self.skipped_nonfinite_steps),
        }

    def training_step(
        self,
        batch: dict[str, Any],
        collect_metrics: bool = False,
        metrics_prefix: str = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        input_latents, timesteps, target_eps = self._latent_batch_to_device(batch)
        prompt_embeds, pooled_prompt_embeds, add_time_ids = self._encode_prompts(batch["prompt"])

        noise_pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
            return_dict=False,
        )[0]
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target_eps.float())
        if not collect_metrics:
            return loss, {}

        metrics: dict[str, float] = {}
        with torch.no_grad():
            if self.cfg.logging.timestep_stats:
                inversion_steps = batch["inversion_step_idx"].float()
                metrics.update(
                    {
                        f"{metrics_prefix}/timestep_mean": float(
                            timesteps.float().mean().detach().cpu()
                        ),
                        f"{metrics_prefix}/timestep_min": float(timesteps.min().detach().cpu()),
                        f"{metrics_prefix}/timestep_max": float(timesteps.max().detach().cpu()),
                        f"{metrics_prefix}/inversion_step_mean": float(
                            inversion_steps.mean().cpu()
                        ),
                        f"{metrics_prefix}/inversion_step_min": float(inversion_steps.min().cpu()),
                        f"{metrics_prefix}/inversion_step_max": float(inversion_steps.max().cpu()),
                    }
                )

            if self.cfg.logging.prediction_stats:
                pred_flat = noise_pred.float().flatten(start_dim=1)
                target_flat = target_eps.float().flatten(start_dim=1)
                cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
                metrics.update(
                    {
                        f"{metrics_prefix}/pred_mean": float(
                            noise_pred.float().mean().detach().cpu()
                        ),
                        f"{metrics_prefix}/pred_std": float(
                            noise_pred.float().std().detach().cpu()
                        ),
                        f"{metrics_prefix}/target_mean": float(
                            target_eps.float().mean().detach().cpu()
                        ),
                        f"{metrics_prefix}/target_std": float(
                            target_eps.float().std().detach().cpu()
                        ),
                        f"{metrics_prefix}/pred_target_mae": float(
                            torch.nn.functional.l1_loss(noise_pred.float(), target_eps.float())
                            .detach()
                            .cpu()
                        ),
                        f"{metrics_prefix}/pred_target_cosine": float(
                            cosine.mean().detach().cpu()
                        ),
                    }
                )
        return loss, metrics

    def _build_step_log(
        self,
        loss_value: float,
        grad_norm_value: float,
        step_metrics: dict[str, float],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self.cfg.logging.basic:
            metrics.update(
                {
                    "train/loss_step": loss_value,
                    "train/global_step": self.global_step,
                    "train/skipped_nonfinite_steps": self.skipped_nonfinite_steps,
                }
            )

        if self.cfg.logging.loss_log10:
            metrics["train/loss_log10"] = self._safe_log10(loss_value)

        if self.cfg.logging.gradients:
            metrics["train/grad_norm"] = grad_norm_value

        if self.cfg.logging.lora:
            metrics["train/lora_param_norm"] = self._param_norm()

        if self.cfg.logging.learning_rate:
            metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        metrics.update(step_metrics)
        return metrics

    @torch.no_grad()
    def validation_epoch(self, val_loader: DataLoader, epoch: int) -> dict[str, float]:
        self.unet.eval()
        losses: list[float] = []
        early_losses: list[float] = []
        other_losses: list[float] = []
        metric_values: dict[str, list[float]] = {}

        for batch in tqdm(val_loader, desc=f"Validation epoch {epoch + 1}", leave=False):
            loss, step_metrics = self.training_step(
                batch,
                collect_metrics=bool(
                    self.cfg.logging.timestep_stats or self.cfg.logging.prediction_stats
                ),
                metrics_prefix="val",
            )
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                continue

            losses.append(loss_value)
            if self._is_early_inversion_batch(batch):
                early_losses.append(loss_value)
            else:
                other_losses.append(loss_value)

            for key, value in step_metrics.items():
                if math.isfinite(value):
                    metric_values.setdefault(key, []).append(value)

        self.unet.train()
        val_loss = self._mean_or_nan(losses)
        metrics = {
            "val/loss": val_loss,
            "val/loss_log10": self._safe_log10(val_loss),
            "val/early_inversion_loss": self._mean_or_nan(early_losses),
            "val/other_inversion_loss": self._mean_or_nan(other_losses),
            "epoch": float(epoch + 1),
            "global_step": float(self.global_step),
        }
        for key, values in metric_values.items():
            metrics[key] = sum(values) / len(values)
        return metrics

    def _latent_batch_to_device(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.unet.dtype
        return (
            batch["input_latent"].to(device=self.device, dtype=dtype),
            batch["timestep"].to(device=self.device),
            batch["target_eps"].to(device=self.device, dtype=dtype),
        )

    @torch.no_grad()
    def _encode_prompts(
        self, prompts: list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = encode_prompt_sdxl(
            pipe=self.pipe,
            prompt=list(prompts),
            negative_prompt=[""] * len(prompts),
            height=int(self.cfg.model.height),
            width=int(self.cfg.model.width),
            do_classifier_free_guidance=False,
        )
        return (
            cond["prompt_embeds"].to(device=self.device, dtype=self.unet.dtype),
            cond["pooled_prompt_embeds"].to(device=self.device, dtype=self.unet.dtype),
            cond["add_time_ids"].to(device=self.device, dtype=self.unet.dtype),
        )

    def _log(self, metrics: dict[str, float]) -> None:
        logger.info("{}", metrics)
        if self.tracker is not None:
            self.tracker.log(metrics, step=self.global_step)

    def _is_early_inversion_batch(self, batch: dict[str, Any]) -> bool:
        early_steps = torch.ceil(
            batch["num_steps"].float() * float(self.cfg.data.early_inversion_fraction)
        )
        return bool(torch.all(batch["inversion_step_idx"].float() < early_steps).item())

    def _handle_nonfinite_batch(
        self,
        reason: str,
        value: float,
        step_metrics: dict[str, float],
    ) -> None:
        self.skipped_nonfinite_steps += 1
        metrics = {
            "train/nonfinite_batch": 1.0,
            "train/nonfinite_value": value,
            "train/skipped_nonfinite_steps": float(self.skipped_nonfinite_steps),
            "train/global_step": float(self.global_step),
            **step_metrics,
        }
        logger.warning(
            "Skipping non-finite {}={} at global_step={}", reason, value, self.global_step
        )
        self._log(metrics)

        if not bool(self.cfg.training.skip_nonfinite_batches):
            raise FloatingPointError(f"Non-finite training {reason}: {value}")

        max_skips = int(self.cfg.training.max_nonfinite_batches)
        if max_skips >= 0 and self.skipped_nonfinite_steps > max_skips:
            raise FloatingPointError(
                f"Exceeded max_nonfinite_batches={max_skips}; latest {reason}={value}"
            )

    def _grad_norm(self) -> float:
        squared_norm = 0.0
        for param in self.trainable_params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            squared_norm += float(torch.sum(grad * grad).cpu())
        return math.sqrt(squared_norm)

    def _param_norm(self) -> float:
        squared_norm = 0.0
        for param in self.trainable_params:
            value = param.detach().float()
            squared_norm += float(torch.sum(value * value).cpu())
        return math.sqrt(squared_norm)

    @staticmethod
    def _safe_log10(value: float) -> float:
        return math.log10(max(value, 1e-20))

    @staticmethod
    def _mean_or_nan(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def save_checkpoint(self, filename: str) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / filename
        torch.save(
            get_peft_model_state_dict(self.unet, adapter_name=self.adapter_name),
            save_path,
        )
        logger.success("Saved LoRA checkpoint: {}", save_path)


def _make_loader(dataset: LatentTrajectoryDataset, cfg: DictConfig, shuffle: bool) -> DataLoader:
    sampler = None
    if shuffle and bool(cfg.oversample_early_inversion_steps):
        weights = dataset.early_inversion_weights(
            early_fraction=float(cfg.early_inversion_fraction),
            early_weight=float(cfg.early_inversion_weight),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info(
            "Using weighted sampler: first {:.1%} inversion steps have {:.2f}x weight",
            float(cfg.early_inversion_fraction),
            float(cfg.early_inversion_weight),
        )

    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )


@hydra.main(config_path="../../config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training SDXL inversion LoRA for {}", cfg.model.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("This training script is intended to run on CUDA.")

    torch.manual_seed(int(cfg.training.seed))
    pipe = make_pipe(cfg.model, device)

    train_dataset = LatentTrajectoryDataset(
        root_dir=_resolve_path(cfg.data.root_dir),
        latents_dir_name=str(cfg.data.latents_dir_name),
        pred_noises_dir_name=str(cfg.data.pred_noises_dir_name),
    )
    val_dataset = None
    if cfg.data.val_root_dir:
        val_dataset = LatentTrajectoryDataset(
            root_dir=_resolve_path(cfg.data.val_root_dir),
            latents_dir_name=str(cfg.data.latents_dir_name),
            pred_noises_dir_name=str(cfg.data.pred_noises_dir_name),
        )

    logger.info("Train items: {}", len(train_dataset))
    if val_dataset is not None:
        logger.info("Validation items: {}", len(val_dataset))

    tracker = _init_wandb(cfg)
    trainer = SDXLInversionLoraTrainer(pipe=pipe, cfg=cfg, tracker=tracker)
    trainer.fit(
        train_loader=_make_loader(train_dataset, cfg.data, shuffle=True),
        val_loader=_make_loader(val_dataset, cfg.data, shuffle=False) if val_dataset else None,
    )


if __name__ == "__main__":
    main()
