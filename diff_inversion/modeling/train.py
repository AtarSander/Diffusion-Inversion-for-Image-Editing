import re
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.utils import make_pipe

SINGLE_ADAPTER_NAME = "inversion"
PAIR_CONDITIONAL_ADAPTER_NAME = "text_branch"
PAIR_UNCONDITIONAL_ADAPTER_NAME = "null_branch"

CFG_LOSS_MODES = frozenset({"cfg", "cfg_pair"})
CFG_SINGLE_PASS_MODES = frozenset({"cfg_single_pass"})
CFG_PAIR_MODES = frozenset({"cfg_pair"})
SHARED_BRANCH_LOSS_MODES = frozenset({"branch_pair_shared"})
PAIR_BRANCH_LOSS_MODES = frozenset({"branch_pair"})
PAIR_ADAPTER_MODES = CFG_PAIR_MODES | PAIR_BRANCH_LOSS_MODES
CFG_REQUIRED_MODES = (
    CFG_LOSS_MODES
    | CFG_SINGLE_PASS_MODES
    | SHARED_BRANCH_LOSS_MODES
    | PAIR_BRANCH_LOSS_MODES
)
TRAINING_TARGET_MODES = CFG_REQUIRED_MODES | {"conditional", "unconditional"}
CFG_BRANCH_TARGET_MODES = CFG_REQUIRED_MODES | {"unconditional"}
NEGATIVE_CONDITIONING_MODES = (
    CFG_LOSS_MODES
    | SHARED_BRANCH_LOSS_MODES
    | PAIR_BRANCH_LOSS_MODES
    | {"unconditional"}
)


class SDXLInversionTrainer:
    def __init__(
        self,
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
        lora_config: LoraConfig,
        tracker: Any,
        checkpoint_dir: Path | str,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler_config: DictConfig | None,
        gradient_accumulation_steps: int,
        num_inference_steps: int,
        height: int,
        width: int,
        save_every_steps: int,
        eval_every_steps: int,
        log_every_steps: int,
        max_val_batches: int | None,
        max_grad_norm: float | None = None,
        gradient_checkpointing: bool = False,
        validation_preview_config: DictConfig | None = None,
        training_target_mode: str = "conditional",
        training_guidance_scale: float | None = None,
    ):
        self.pipe = pipe
        self.lora_config = lora_config
        self.model = pipe.unet
        self.tracker = tracker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.save_every_steps = int(save_every_steps)
        self.eval_every_steps = int(eval_every_steps)
        self.log_every_steps = int(log_every_steps)
        self.max_val_batches = max_val_batches
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler = None
        self._pending_lr_scheduler_state = None
        self.validation_preview_config = validation_preview_config
        self.height = int(height)
        self.width = int(width)
        self.training_target_mode = self._normalize_training_target_mode(training_target_mode)
        self.training_guidance_scale = (
            None if training_guidance_scale is None else float(training_guidance_scale)
        )
        if (
            self.training_target_mode in CFG_REQUIRED_MODES
            and self.training_guidance_scale is not None
            and self.training_guidance_scale <= 1.0
        ):
            raise ValueError(
                f"training_target.mode={self.training_target_mode} requires "
                "training_target.guidance_scale > 1.0 "
                f"or null per-sample guidance; got {self.training_guidance_scale}."
            )
        self._freeze_pipeline_components()
        self._inject_lora_adapters(lora_config)
        self._freeze_non_lora_parameters()
        self._cast_trainable_parameters(torch.float32)
        if self._uses_pair_adapters():
            self._set_active_adapter(PAIR_CONDITIONAL_ADAPTER_NAME)
        else:
            self._set_active_adapter(SINGLE_ADAPTER_NAME)

        if gradient_checkpointing and hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable_parameters:
            raise RuntimeError("No LoRA parameters were marked trainable.")
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.model.device)
        logger.info(
            "LoRA trainable parameters: {:,}",
            sum(p.numel() for p in self.trainable_parameters),
        )
        logger.info(
            "Training target mode: {} guidance_scale={}",
            self.training_target_mode,
            self.training_guidance_scale,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_train_steps: int,
        initial_global_step: int = 0,
        save_training_state: bool = False,
    ):
        self.global_step = int(initial_global_step)
        micro_step = 0
        recent_metrics: list[dict[str, float]] = []
        self.lr_scheduler = self._build_lr_scheduler(max_train_steps)
        if self._pending_lr_scheduler_state is not None:
            if self.lr_scheduler is None:
                raise RuntimeError(
                    "Cannot restore LR scheduler state because no scheduler is configured."
                )
            self.lr_scheduler.load_state_dict(self._pending_lr_scheduler_state)
            logger.info(
                "Restored LR scheduler state at global_step={} lr={:.8g}",
                self.global_step,
                self.optimizer.param_groups[0]["lr"],
            )
        else:
            self._advance_lr_scheduler(self.global_step)

        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=max_train_steps, initial=self.global_step, desc="Training steps")

        while self.global_step < max_train_steps:
            for batch in train_loader:
                loss, metrics = self.forward_loss_with_metrics(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(self._non_finite_loss_message(batch, loss))

                (loss / self.gradient_accumulation_steps).backward()
                recent_metrics.append(self._detach_metrics(metrics))
                micro_step += 1

                if micro_step % self.gradient_accumulation_steps != 0:
                    continue

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                progress.update(1)

                if self._should_run(self.log_every_steps):
                    train_metrics = self._average_metric_rows(recent_metrics, prefix="train")
                    train_metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
                    self.tracker.log(train_metrics, step=self.global_step)
                    recent_metrics = []

                if self._should_run(self.eval_every_steps):
                    self.tracker.log(
                        self.validation_epoch(val_loader),
                        step=self.global_step,
                    )
                    self._run_validation_preview()

                if self._should_run(self.save_every_steps):
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                    if save_training_state:
                        self.save_training_state(f"training_state_step_{self.global_step}.pt")

                if self.global_step >= max_train_steps:
                    break

        progress.close()
        self.save_checkpoint("checkpoint_final.pt")
        if save_training_state:
            self.save_training_state("training_state_final.pt")
        self.tracker.finish()

    def forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        loss, _ = self.forward_loss_with_metrics(batch)
        return loss

    def forward_loss_with_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_clean = batch["x_clean"].to(device=self.model.device, dtype=self.model.dtype)
        timestep = batch["timestep"].to(device=self.model.device)
        target_eps = batch["target_eps"].to(device=self.model.device, dtype=self.model.dtype)
        prompt_embeds = batch["prompt_embeds"].to(
            device=self.model.device,
            dtype=self.model.dtype,
        )
        pooled_prompt_embeds = batch.get("pooled_prompt_embeds")
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                device=self.model.device,
                dtype=self.model.dtype,
            )
        add_time_ids = batch.get("add_time_ids")
        if add_time_ids is not None:
            add_time_ids = add_time_ids.to(device=self.model.device, dtype=self.model.dtype)

        scheduler_timesteps = self._scheduler_timesteps(timestep, batch_size=x_clean.shape[0])

        target_eps_uncond = None
        if self.training_target_mode in CFG_BRANCH_TARGET_MODES:
            target_eps_uncond = self._required_tensor(batch, "target_eps_uncond").to(
                device=self.model.device,
                dtype=self.model.dtype,
            )

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if self.training_target_mode in NEGATIVE_CONDITIONING_MODES:
            negative_prompt_embeds = self._required_tensor(batch, "negative_prompt_embeds").to(
                device=self.model.device,
                dtype=self.model.dtype,
            )
            negative_pooled_prompt_embeds = batch.get("negative_pooled_prompt_embeds")
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
            elif pooled_prompt_embeds is not None:
                raise KeyError(
                    "negative_pooled_prompt_embeds are required for SDXL "
                    f"training_target.mode={self.training_target_mode}."
                )

        guidance_scale = None
        if self.training_target_mode in CFG_REQUIRED_MODES:
            sample_guidance_scale = None
            if self.training_guidance_scale is None:
                sample_guidance_scale = self._required_tensor(batch, "sample_guidance_scale")
            guidance_scale = self._guidance_scale(
                sample_guidance_scale,
                batch_size=x_clean.shape[0],
                device=self.model.device,
            )

        if self.training_target_mode in CFG_LOSS_MODES:
            assert target_eps_uncond is not None
            assert negative_prompt_embeds is not None
            assert guidance_scale is not None
            return self._forward_cfg_loss(
                x_clean=x_clean,
                scheduler_timesteps=scheduler_timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                target_eps=target_eps,
                target_eps_uncond=target_eps_uncond,
                guidance_scale=guidance_scale,
            )
        if self.training_target_mode in CFG_SINGLE_PASS_MODES:
            assert target_eps_uncond is not None
            assert guidance_scale is not None
            return self._forward_cfg_single_pass_loss(
                x_clean=x_clean,
                scheduler_timesteps=scheduler_timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                target_eps=target_eps,
                target_eps_uncond=target_eps_uncond,
                guidance_scale=guidance_scale,
            )
        if self.training_target_mode == "unconditional":
            assert target_eps_uncond is not None
            assert negative_prompt_embeds is not None
            return self._forward_unconditional_loss(
                x_clean=x_clean,
                scheduler_timesteps=scheduler_timesteps,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                target_eps_uncond=target_eps_uncond,
            )
        if self.training_target_mode in SHARED_BRANCH_LOSS_MODES:
            assert target_eps_uncond is not None
            assert negative_prompt_embeds is not None
            assert guidance_scale is not None
            return self._forward_shared_branch_pair_loss(
                x_clean=x_clean,
                scheduler_timesteps=scheduler_timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                target_eps=target_eps,
                target_eps_uncond=target_eps_uncond,
                guidance_scale=guidance_scale,
            )
        if self._uses_pair_adapters():
            assert target_eps_uncond is not None
            assert negative_prompt_embeds is not None
            assert guidance_scale is not None
            return self._forward_branch_pair_loss(
                x_clean=x_clean,
                scheduler_timesteps=scheduler_timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                target_eps=target_eps,
                target_eps_uncond=target_eps_uncond,
                guidance_scale=guidance_scale,
            )

        student_eps = self.predict_noise(
            x_clean,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )

        loss = F.mse_loss(student_eps.float(), target_eps.float())
        return loss, {
            "loss": loss,
            "loss_cond": loss,
        }

    def _forward_unconditional_loss(
        self,
        *,
        x_clean: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
        target_eps_uncond: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_eps = self.predict_noise(
            x_clean,
            scheduler_timesteps,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            add_time_ids,
        )
        loss = F.mse_loss(student_eps.float(), target_eps_uncond.float())
        return loss, {
            "loss": loss,
            "loss_uncond": loss,
        }

    def _forward_shared_branch_pair_loss(
        self,
        *,
        x_clean: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
        target_eps: torch.Tensor,
        target_eps_uncond: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_cond = self.predict_noise_with_adapter(
            SINGLE_ADAPTER_NAME,
            x_clean,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )
        pred_uncond = self.predict_noise_with_adapter(
            SINGLE_ADAPTER_NAME,
            x_clean,
            scheduler_timesteps,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            add_time_ids,
        )
        loss_cond = F.mse_loss(pred_cond.float(), target_eps.float())
        loss_uncond = F.mse_loss(pred_uncond.float(), target_eps_uncond.float())
        guidance_scale = guidance_scale.reshape(x_clean.shape[0], *([1] * (target_eps.ndim - 1)))
        pred_cfg = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        target_cfg = target_eps_uncond + guidance_scale * (target_eps - target_eps_uncond)
        loss_cfg = F.mse_loss(pred_cfg.float(), target_cfg.float())

        loss = loss_cond + loss_uncond

        return loss, {
            "loss": loss,
            "loss_cond": loss_cond,
            "loss_uncond": loss_uncond,
            "loss_cfg": loss_cfg,
        }

    def _forward_branch_pair_loss(
        self,
        *,
        x_clean: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
        target_eps: torch.Tensor,
        target_eps_uncond: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_cond = self.predict_noise_with_adapter(
            PAIR_CONDITIONAL_ADAPTER_NAME,
            x_clean,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )
        pred_uncond = self.predict_noise_with_adapter(
            PAIR_UNCONDITIONAL_ADAPTER_NAME,
            x_clean,
            scheduler_timesteps,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            add_time_ids,
        )
        loss_cond = F.mse_loss(pred_cond.float(), target_eps.float())
        loss_uncond = F.mse_loss(pred_uncond.float(), target_eps_uncond.float())

        guidance_scale = guidance_scale.reshape(
            x_clean.shape[0],
            *([1] * (target_eps.ndim - 1)),
        )
        pred_cfg = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        target_cfg = target_eps_uncond + guidance_scale * (target_eps - target_eps_uncond)
        loss_cfg = F.mse_loss(pred_cfg.float(), target_cfg.float())

        loss = loss_cond + loss_uncond

        return loss, {
            "loss": loss,
            "loss_cond": loss_cond,
            "loss_uncond": loss_uncond,
            "loss_cfg": loss_cfg,
        }

    def _forward_cfg_loss(
        self,
        *,
        x_clean: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
        target_eps: torch.Tensor,
        target_eps_uncond: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_uncond, pred_cond = self.predict_noise_cfg_branches(
            x_clean,
            scheduler_timesteps,
            negative_prompt_embeds,
            prompt_embeds,
            negative_pooled_prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )
        guidance_scale = guidance_scale.reshape(
            x_clean.shape[0],
            *([1] * (target_eps.ndim - 1)),
        )

        pred_cfg = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        target_cfg = target_eps_uncond + guidance_scale * (target_eps - target_eps_uncond)

        loss_cfg = F.mse_loss(pred_cfg.float(), target_cfg.float())
        loss_cond = F.mse_loss(pred_cond.float(), target_eps.float())
        loss_uncond = F.mse_loss(pred_uncond.float(), target_eps_uncond.float())
        return loss_cfg, {
            "loss": loss_cfg,
            "loss_cfg": loss_cfg,
            "loss_cond": loss_cond,
            "loss_uncond": loss_uncond,
        }

    def _forward_cfg_single_pass_loss(
        self,
        *,
        x_clean: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
        target_eps: torch.Tensor,
        target_eps_uncond: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Learn the combined CFG target with one conditional UNet prediction."""
        guidance_scale_broadcast = guidance_scale.reshape(
            x_clean.shape[0],
            *([1] * (target_eps.ndim - 1)),
        )
        target_cfg = target_eps_uncond + guidance_scale_broadcast * (
            target_eps - target_eps_uncond
        )

        pred_cfg = self.predict_noise(
            x_clean,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )
        loss = F.mse_loss(pred_cfg.float(), target_cfg.float())
        return loss, {
            "loss": loss,
            "loss_cfg_single_pass": loss,
        }

    def predict_noise(
        self,
        latents: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        model_input = self.pipe.scheduler.scale_model_input(latents, scheduler_timesteps)
        unet_kwargs = {}
        if pooled_prompt_embeds is not None and add_time_ids is not None:
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }
        return self.model(
            model_input,
            scheduler_timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            **unet_kwargs,
        )[0]

    def predict_noise_with_adapter(
        self,
        adapter_name: str,
        latents: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        self._set_active_adapter(adapter_name)
        return self.predict_noise(
            latents,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )

    def predict_noise_cfg_branches(
        self,
        latents: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_target_mode in CFG_PAIR_MODES:
            if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
                raise KeyError("negative_pooled_prompt_embeds are required for SDXL CFG training.")
            pred_uncond = self.predict_noise_with_adapter(
                PAIR_UNCONDITIONAL_ADAPTER_NAME,
                latents,
                scheduler_timesteps,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                add_time_ids,
            )
            pred_cond = self.predict_noise_with_adapter(
                PAIR_CONDITIONAL_ADAPTER_NAME,
                latents,
                scheduler_timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                add_time_ids,
            )
            return pred_uncond, pred_cond

        branch_latents = torch.cat([latents, latents], dim=0)
        branch_timesteps = torch.cat([scheduler_timesteps, scheduler_timesteps], dim=0)
        branch_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        branch_pooled_prompt_embeds = None
        if pooled_prompt_embeds is not None:
            if negative_pooled_prompt_embeds is None:
                raise KeyError("negative_pooled_prompt_embeds are required for SDXL CFG training.")
            branch_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )
        branch_add_time_ids = None
        if add_time_ids is not None:
            branch_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        noise_pred = self.predict_noise(
            branch_latents,
            branch_timesteps,
            branch_prompt_embeds,
            branch_pooled_prompt_embeds,
            branch_add_time_ids,
        )
        noise_uncond, noise_cond = noise_pred.chunk(2)
        return noise_uncond, noise_cond

    @torch.no_grad()
    def validation_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        metric_rows: list[dict[str, float]] = []

        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break
            loss, metrics = self.forward_loss_with_metrics(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(self._non_finite_loss_message(batch, loss, "validation"))
            metric_rows.append(self._detach_metrics(metrics))

        self.model.train()
        if not metric_rows:
            return {"val/loss": float("nan")}
        return self._average_metric_rows(metric_rows, prefix="val")

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        device = getattr(self.pipe, "_execution_device", self.model.device)
        if getattr(self.pipe, "text_encoder_2", None) is not None:
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt_list,
                prompt_2=prompt_list,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            add_time_ids = self.pipe._get_add_time_ids(
                original_size=(self.height, self.width),
                crops_coords_top_left=(0, 0),
                target_size=(self.height, self.width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim,
            ).to(device=self.model.device)
            add_time_ids = add_time_ids.repeat(len(prompt_list), 1)

            return (
                prompt_embeds.to(device=self.model.device, dtype=self.model.dtype),
                pooled_prompt_embeds.to(device=self.model.device, dtype=self.model.dtype),
                add_time_ids,
            )

        encoded = self.pipe.encode_prompt(
            prompt=prompt_list,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = encoded[0] if isinstance(encoded, tuple) else encoded
        return prompt_embeds.to(device=self.model.device, dtype=self.model.dtype), None, None

    def save_checkpoint(self, filename: str):
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self._uses_pair_adapters():
            state = {
                "branch_pair": True,
                "adapters": {
                    "conditional": self._adapter_state_dict(PAIR_CONDITIONAL_ADAPTER_NAME),
                    "unconditional": self._adapter_state_dict(PAIR_UNCONDITIONAL_ADAPTER_NAME),
                },
            }
            torch.save(state, save_path)
        else:
            torch.save(self._single_adapter_state_dict(), save_path)
        logger.info("Checkpoint saved to {}", save_path)

    def save_training_state(self, filename: str):
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        if self._uses_pair_adapters():
            state["lora_state_dicts"] = {
                "conditional": self._adapter_state_dict(PAIR_CONDITIONAL_ADAPTER_NAME),
                "unconditional": self._adapter_state_dict(PAIR_UNCONDITIONAL_ADAPTER_NAME),
            }
        else:
            state["lora_state_dict"] = self._single_adapter_state_dict()
        torch.save(state, save_path)
        logger.info("Training state saved to {}", save_path)

    def load_checkpoint(self, filename: str | Path):
        checkpoint_path = Path(filename)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self._uses_pair_adapters():
            self._load_branch_pair_checkpoint_state(state_dict, checkpoint_path)
            logger.info("Checkpoint loaded from {}", checkpoint_path)
            return
        self._load_single_adapter_state_dict(state_dict, checkpoint_path)
        logger.info("Checkpoint loaded from {}", checkpoint_path)

    def load_training_state(self, filename: str | Path) -> int:
        checkpoint_path = Path(filename)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Training state checkpoint does not exist: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Not a full training-state checkpoint: {checkpoint_path}")

        if self._uses_pair_adapters():
            if "lora_state_dicts" not in state:
                raise ValueError(f"Not a branch-pair training-state checkpoint: {checkpoint_path}")
            self._load_branch_pair_state_dicts(state["lora_state_dicts"], checkpoint_path)
        else:
            if "lora_state_dict" not in state:
                raise ValueError(f"Not a full training-state checkpoint: {checkpoint_path}")
            self._load_single_adapter_state_dict(state["lora_state_dict"], checkpoint_path)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._pending_lr_scheduler_state = state["lr_scheduler_state_dict"]
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        global_step = int(state["global_step"])
        logger.info(
            "Training state loaded from {} at global_step={}", checkpoint_path, global_step
        )
        return global_step

    def _inject_lora_adapters(self, lora_config: LoraConfig) -> None:
        if self._uses_pair_adapters():
            inject_adapter_in_model(
                lora_config,
                self.model,
                adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME,
            )
            inject_adapter_in_model(
                lora_config,
                self.model,
                adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME,
            )
        else:
            inject_adapter_in_model(lora_config, self.model, adapter_name=SINGLE_ADAPTER_NAME)

    def _single_adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return get_peft_model_state_dict(
            self.model,
            adapter_name=SINGLE_ADAPTER_NAME,
        )

    def _load_single_adapter_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        checkpoint_path: Path,
    ) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError(f"Not a LoRA checkpoint state dict: {checkpoint_path}")
        set_peft_model_state_dict(self.model, state_dict, adapter_name=SINGLE_ADAPTER_NAME)

    def _adapter_state_dict(self, adapter_name: str) -> dict[str, torch.Tensor]:
        needle = f".{adapter_name}."
        state: dict[str, torch.Tensor] = {}
        for key, value in self.model.state_dict().items():
            if "lora" not in key.lower() or needle not in key:
                continue
            state[key.replace(needle, ".")] = value.detach().cpu()
        if not state:
            raise RuntimeError(f"No LoRA state found for adapter {adapter_name!r}.")
        return state

    def _load_branch_pair_checkpoint_state(
        self,
        state: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        if not isinstance(state, dict) or state.get("branch_pair") is not True:
            raise ValueError(f"Not a branch-pair checkpoint: {checkpoint_path}")
        adapters = state.get("adapters")
        if not isinstance(adapters, dict):
            raise ValueError(f"Branch-pair checkpoint is missing adapters: {checkpoint_path}")
        self._load_branch_pair_state_dicts(adapters, checkpoint_path)

    def _load_branch_pair_state_dicts(
        self,
        state_dicts: dict[str, dict[str, torch.Tensor]],
        checkpoint_path: Path,
    ) -> None:
        if "conditional" not in state_dicts or "unconditional" not in state_dicts:
            raise ValueError(
                "Branch-pair checkpoint must contain 'conditional' and "
                f"'unconditional' adapter states: {checkpoint_path}"
            )
        set_peft_model_state_dict(
            self.model,
            state_dicts["conditional"],
            adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME,
        )
        set_peft_model_state_dict(
            self.model,
            state_dicts["unconditional"],
            adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME,
        )
        self._set_lora_parameters_trainable()

    def _freeze_pipeline_components(self) -> None:
        for component in (
            self.pipe.text_encoder,
            getattr(self.pipe, "text_encoder_2", None),
            self.pipe.vae,
        ):
            if component is not None:
                component.requires_grad_(False)
                component.eval()

    def _freeze_non_lora_parameters(self) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora" in name.lower())

    def _set_lora_parameters_trainable(self) -> None:
        for name, parameter in self.model.named_parameters():
            if "lora" in name.lower():
                parameter.requires_grad_(True)

    def _cast_trainable_parameters(self, dtype: torch.dtype) -> None:
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=dtype)

    def _build_lr_scheduler(self, max_train_steps: int):
        if self.lr_scheduler_config is None:
            return None

        scheduler_name = str(self.lr_scheduler_config.name)
        warmup_steps = int(self.lr_scheduler_config.warmup_steps)
        num_cycles = int(self.lr_scheduler_config.num_cycles)
        power = float(self.lr_scheduler_config.power)
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=num_cycles,
            power=power,
        )
        logger.info(
            "Using LR scheduler: {} warmup_steps={} num_training_steps={}",
            scheduler_name,
            warmup_steps,
            max_train_steps,
        )
        return scheduler

    def _advance_lr_scheduler(self, global_step: int) -> None:
        if self.lr_scheduler is None or global_step <= 0:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
            )
            for _ in range(global_step):
                self.lr_scheduler.step()

        logger.info(
            "Advanced LR scheduler to global_step={} lr={:.8g}",
            global_step,
            self.optimizer.param_groups[0]["lr"],
        )

    def _non_finite_loss_message(
        self,
        batch: dict[str, Any],
        loss: torch.Tensor,
        phase: str = "training",
    ) -> str:
        sample_idx = self._batch_scalar(batch.get("sample_idx"))
        step_idx = self._batch_scalar(batch.get("step_idx"))
        timestep = self._batch_scalar(batch.get("timestep"))
        return (
            f"Non-finite {phase} loss "
            f"loss={float(loss.detach().cpu())} "
            f"global_step={getattr(self, 'global_step', 'unknown')} "
            f"sample_idx={sample_idx} step_idx={step_idx} timestep={timestep}. "
            "Stopping before logging NaNs to W&B."
        )

    @staticmethod
    def _batch_scalar(value: Any) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.flatten()
            return value[0].item() if value.numel() else None
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    @staticmethod
    def _required_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise KeyError(f"Batch is missing required tensor {key!r}.")
        return value

    def _guidance_scale(
        self,
        sample_guidance_scale: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        if self.training_guidance_scale is not None:
            guidance_scale = torch.full(
                (batch_size,),
                self.training_guidance_scale,
                device=device,
                dtype=torch.float32,
            )
        else:
            if sample_guidance_scale is None:
                raise KeyError("Batch is missing required tensor 'sample_guidance_scale'.")
            guidance_scale = sample_guidance_scale.to(
                device=device,
                dtype=torch.float32,
            ).flatten()
            if guidance_scale.numel() == 1:
                guidance_scale = guidance_scale.expand(batch_size)
            if guidance_scale.numel() != batch_size:
                raise ValueError(
                    "sample_guidance_scale batch shape does not match latent batch shape: "
                    f"got {guidance_scale.numel()} values for batch_size={batch_size}."
                )

        if torch.any(guidance_scale <= 1.0):
            min_guidance = float(guidance_scale.detach().float().min().cpu())
            raise ValueError(
                "CFG training modes require guidance_scale > 1.0; "
                f"got minimum batch value {min_guidance}."
            )
        return guidance_scale

    @staticmethod
    def _detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            key: float(value.detach().float().cpu())
            for key, value in metrics.items()
            if torch.is_tensor(value) and torch.isfinite(value.detach()).all().item()
        }

    @staticmethod
    def _average_metric_rows(
        rows: list[dict[str, float]],
        *,
        prefix: str,
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            for key, value in row.items():
                grouped.setdefault(key, []).append(float(value))
        return {
            f"{prefix}/{key}": sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }

    @staticmethod
    def _normalize_training_target_mode(value: str) -> str:
        mode = str(value).strip().lower()
        if mode == "cfg_distill":
            warnings.warn(
                "training_target.mode=cfg_distill is deprecated; use cfg_single_pass.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "cfg_single_pass"
        if mode not in TRAINING_TARGET_MODES:
            raise ValueError(
                "training_target.mode must be 'conditional', 'unconditional', "
                "'cfg', 'cfg_single_pass', 'cfg_pair', 'branch_pair_shared', or "
                f"'branch_pair', got {value!r}."
            )
        return mode

    def _uses_pair_adapters(self) -> bool:
        return self.training_target_mode in PAIR_ADAPTER_MODES

    def _set_active_adapter(self, adapter_name: str) -> None:
        toggled = False
        for module in self.model.modules():
            if module is self.model:
                continue
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_name)
                toggled = True

        if not toggled:
            raise AttributeError("No injected LoRA adapter layers expose set_adapter().")

        # PEFT marks only the active adapter trainable. Branch-pair training needs
        # gradients for both adapter graphs after two forward passes.
        self._set_lora_parameters_trainable()

    def _scheduler_timesteps(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        timestep = timestep.flatten()
        if timestep.numel() == 1:
            return timestep.expand(batch_size)
        if timestep.numel() != batch_size:
            raise ValueError(
                "Timestep batch shape does not match latent batch shape: "
                f"got {timestep.numel()} timesteps for batch_size={batch_size}."
            )
        return timestep

    def _should_run(self, interval: int) -> bool:
        return interval > 0 and self.global_step > 0 and self.global_step % interval == 0

    def _run_validation_preview(self) -> None:
        if self.validation_preview_config is None:
            return
        if self.training_target_mode in PAIR_ADAPTER_MODES:
            logger.warning(
                "Validation preview skipped for two-adapter training mode {}; "
                "two-adapter inversion preview is not implemented.",
                self.training_target_mode,
            )
            return
        if self.training_target_mode in SHARED_BRANCH_LOSS_MODES:
            logger.warning(
                "Validation preview skipped for shared branch-loss mode {}; "
                "branch-pair preview is not implemented.",
                self.training_target_mode,
            )
            return

        from diff_inversion.modeling.validation_preview import (
            log_validation_preview,
            should_run_validation_preview,
        )

        if not should_run_validation_preview(self.validation_preview_config, self.global_step):
            return

        log_validation_preview(
            pipe=self.pipe,
            cfg=self.validation_preview_config,
            tracker=self.tracker,
            checkpoint_dir=self.checkpoint_dir,
            global_step=self.global_step,
        )


def get_lora_config(lora_config: DictConfig) -> LoraConfig:
    return LoraConfig(**lora_config)


def infer_step_from_checkpoint_path(path: str | Path) -> int | None:
    match = re.search(r"(?:checkpoint|training_state)_step_(\d+)\.pt$", Path(path).name)
    return int(match.group(1)) if match else None


def _resolve_resume_step(resume_cfg: DictConfig, checkpoint_path: Path) -> int:
    configured_step = resume_cfg.global_step
    if configured_step is not None:
        return int(configured_step)

    inferred_step = infer_step_from_checkpoint_path(checkpoint_path)
    if inferred_step is not None:
        return inferred_step

    raise ValueError(
        "resume.global_step is required when the checkpoint filename does not contain "
        "`checkpoint_step_<step>.pt` or `training_state_step_<step>.pt`."
    )


def _dataset_roots(cfg: DictConfig, plural_key: str, singular_key: str):
    """Read one or more dataset roots while keeping old configs compatible."""
    plural_value = OmegaConf.select(cfg, f"data.{plural_key}", default=None)
    if plural_value is not None:
        if isinstance(plural_value, str):
            return plural_value
        roots = [str(root) for root in plural_value]
        if roots:
            return roots

    singular_value = OmegaConf.select(cfg, f"data.{singular_key}", default=None)
    if singular_value is None:
        raise ValueError(
            f"Set data.{singular_key} or data.{plural_key} to at least one dataset root."
        )
    return str(singular_value)


@hydra.main(config_path="../../config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training {}", cfg.model.model_id)
    model_cfg = cfg.model
    lora_cfg = cfg.lora

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training seed: {}", seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_cfg.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(model_cfg, device)
    lora_config = get_lora_config(lora_cfg)
    training_target_mode = SDXLInversionTrainer._normalize_training_target_mode(
        str(cfg.training_target.mode)
    )
    OmegaConf.update(cfg, "training_target.mode", training_target_mode)
    training_guidance_scale = cfg.training_target.guidance_scale
    training_guidance_scale = (
        None if training_guidance_scale is None else float(training_guidance_scale)
    )
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.run_name,
        mode=cfg.wandb.mode,
        id=cfg.wandb.get("id"),
        resume=cfg.wandb.get("resume"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = SDXLInversionTrainer(
        pipe=pipe,
        lora_config=lora_config,
        tracker=run,
        checkpoint_dir=cfg.checkpoint_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_config=cfg.lr_scheduler,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_inference_steps=model_cfg.num_inference_steps,
        height=model_cfg.height,
        width=model_cfg.width,
        save_every_steps=cfg.save_every_steps,
        eval_every_steps=cfg.eval_every_steps,
        log_every_steps=cfg.log_every_steps,
        max_val_batches=cfg.max_val_batches,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        validation_preview_config=cfg if bool(cfg.validation_preview.enabled) else None,
        training_target_mode=training_target_mode,
        training_guidance_scale=training_guidance_scale,
    )

    initial_global_step = 0
    resume_cfg = cfg.resume
    if bool(resume_cfg.enabled):
        checkpoint_path_cfg = resume_cfg.checkpoint_path
        if not checkpoint_path_cfg:
            raise ValueError("resume.checkpoint_path must be set when resume.enabled=true.")

        checkpoint_path = Path(str(checkpoint_path_cfg))
        resume_mode = str(resume_cfg.mode)
        if resume_mode == "training_state":
            initial_global_step = trainer.load_training_state(checkpoint_path)
        elif resume_mode == "adapter":
            trainer.load_checkpoint(checkpoint_path)
            initial_global_step = _resolve_resume_step(resume_cfg, checkpoint_path)
            logger.warning(
                "Adapter-only resume from {} at global_step={}. "
                "Optimizer moments are not restored; LR scheduler will be advanced to this step.",
                checkpoint_path,
                initial_global_step,
            )
        else:
            raise ValueError(
                f"Unknown resume.mode={resume_mode!r}; use 'adapter' or 'training_state'."
            )

        if initial_global_step >= int(cfg.max_train_steps):
            raise ValueError(
                "Resume step is not below max_train_steps: "
                f"resume step={initial_global_step}, max_train_steps={cfg.max_train_steps}."
            )

    train_roots = _dataset_roots(cfg, "root_dirs", "root_dir")
    val_roots = _dataset_roots(cfg, "val_root_dirs", "val_root_dir")
    logger.info("Training dataset roots: {}", train_roots)
    logger.info("Validation dataset roots: {}", val_roots)

    train_dataset = LatentTrajectoryDataset(
        train_roots,
        latents_file_name=cfg.data.latents_file_name,
        conditioning_file_name=cfg.data.conditioning_file_name,
        targets_dir_name=cfg.data.targets_dir_name,
        target_eps_file_name=cfg.data.target_eps_file_name,
        target_uncond_eps_file_name=str(cfg.data.target_uncond_eps_file_name),
        load_cfg_branch_targets=training_target_mode in CFG_BRANCH_TARGET_MODES,
        require_training_cache=cfg.data.require_training_cache,
    )
    val_dataset = LatentTrajectoryDataset(
        val_roots,
        latents_file_name=cfg.data.latents_file_name,
        conditioning_file_name=cfg.data.conditioning_file_name,
        targets_dir_name=cfg.data.targets_dir_name,
        target_eps_file_name=cfg.data.target_eps_file_name,
        target_uncond_eps_file_name=str(cfg.data.target_uncond_eps_file_name),
        load_cfg_branch_targets=training_target_mode in CFG_BRANCH_TARGET_MODES,
        require_training_cache=cfg.data.require_training_cache,
    )
    logger.info("Loaded {:,} train transitions", len(train_dataset))
    logger.info("Loaded {:,} val transitions", len(val_dataset))

    train_generator = torch.Generator()
    train_generator.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    trainer.train(
        train_loader,
        val_loader,
        max_train_steps=cfg.max_train_steps,
        initial_global_step=initial_global_step,
        save_training_state=bool(cfg.save_training_state),
    )


if __name__ == "__main__":
    main()
