from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.utils import make_pipe


class SDXLInversionTrainer:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
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
        self.height = int(height)
        self.width = int(width)

        self._freeze_pipeline_components()
        inject_adapter_in_model(lora_config, self.model, adapter_name="inversion")
        self._freeze_non_lora_parameters()
        self._cast_trainable_parameters(torch.float32)

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

    def train(self, train_loader: DataLoader, val_loader: DataLoader, max_train_steps: int):
        self.global_step = 0
        micro_step = 0
        recent_losses: list[float] = []
        self.lr_scheduler = self._build_lr_scheduler(max_train_steps)

        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=max_train_steps, desc="Training steps")

        while self.global_step < max_train_steps:
            for batch in train_loader:
                loss = self.forward_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(self._non_finite_loss_message(batch, loss))

                (loss / self.gradient_accumulation_steps).backward()
                recent_losses.append(float(loss.detach().cpu()))
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
                    train_loss = sum(recent_losses) / len(recent_losses)
                    self.tracker.log(
                        {
                            "train/loss": train_loss,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                    )
                    recent_losses = []

                if self._should_run(self.eval_every_steps):
                    self.tracker.log(
                        self.validation_epoch(val_loader),
                        step=self.global_step,
                    )

                if self._should_run(self.save_every_steps):
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                if self.global_step >= max_train_steps:
                    break

        progress.close()
        self.save_checkpoint("checkpoint_final.pt")
        self.tracker.finish()

    def forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        x_clean = batch["x_clean"].to(device=self.model.device, dtype=self.model.dtype)
        timestep = batch["timestep"].to(device=self.model.device)
        target_eps = batch["target_eps"].to(device=self.model.device, dtype=self.model.dtype)
        prompt_embeds = batch["prompt_embeds"].to(
            device=self.model.device,
            dtype=self.model.dtype,
        )
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(
            device=self.model.device,
            dtype=self.model.dtype,
        )
        add_time_ids = batch["add_time_ids"].to(device=self.model.device, dtype=self.model.dtype)

        scheduler_timesteps = self._scheduler_timesteps(timestep, batch_size=x_clean.shape[0])

        student_eps = self.predict_noise(
            x_clean,
            scheduler_timesteps,
            prompt_embeds,
            pooled_prompt_embeds,
            add_time_ids,
        )

        return F.mse_loss(student_eps.float(), target_eps.float())

    def predict_noise(
        self,
        latents: torch.Tensor,
        scheduler_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        model_input = self.pipe.scheduler.scale_model_input(latents, scheduler_timesteps)
        return self.model(
            model_input,
            scheduler_timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
            return_dict=False,
        )[0]

    @torch.no_grad()
    def validation_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses = []

        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break
            loss = self.forward_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(self._non_finite_loss_message(batch, loss, "validation"))
            losses.append(float(loss.detach().cpu()))

        self.model.train()
        return {"val/loss": sum(losses) / len(losses)} if losses else {"val/loss": float("nan")}

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt_list,
            prompt_2=prompt_list,
            device=self.pipe._execution_device,
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

    def save_checkpoint(self, filename: str):
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(get_peft_model_state_dict(self.model, adapter_name="inversion"), save_path)
        logger.info("Checkpoint saved to {}", save_path)

    def load_checkpoint(self, filename: str):
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            logger.warning("Checkpoint path does not exist: {}", checkpoint_path)
            return
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(self.model, state_dict, adapter_name="inversion")
        logger.info("Checkpoint loaded from {}", checkpoint_path)

    def _freeze_pipeline_components(self) -> None:
        for component in (self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.vae):
            if component is not None:
                component.requires_grad_(False)
                component.eval()

    def _freeze_non_lora_parameters(self) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora" in name.lower())

    def _cast_trainable_parameters(self, dtype: torch.dtype) -> None:
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=dtype)

    def _build_lr_scheduler(self, max_train_steps: int):
        if self.lr_scheduler_config is None:
            return None

        scheduler_name = str(self.lr_scheduler_config.get("name", "constant"))
        warmup_steps = int(self.lr_scheduler_config.get("warmup_steps", 0))
        num_cycles = int(self.lr_scheduler_config.get("num_cycles", 1))
        power = float(self.lr_scheduler_config.get("power", 1.0))
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

    @contextmanager
    def _teacher_mode(self):
        was_training = self.model.training
        self.model.eval()
        self._set_lora_enabled(False)
        try:
            yield
        finally:
            self._set_lora_enabled(True)
            if was_training:
                self.model.train()

    def _set_lora_enabled(self, enabled: bool) -> None:
        toggled = False
        for module in self.model.modules():
            if module is self.model:
                continue
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled)
                toggled = True

        if not toggled:
            raise AttributeError("No injected LoRA adapter layers expose enable_adapters().")

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


def get_lora_config(lora_config: DictConfig) -> LoraConfig:
    return LoraConfig(**lora_config)


@hydra.main(config_path="../../config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training {}", cfg.model.model_id)
    model_cfg = cfg.model
    lora_cfg = cfg.lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_cfg.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(model_cfg, device)
    lora_config = get_lora_config(lora_cfg)

    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.run_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = SDXLInversionTrainer(
        pipe=pipe,
        lora_config=lora_config,
        tracker=run,
        checkpoint_dir=cfg.checkpoint_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_config=cfg.get("lr_scheduler"),
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
    )

    train_dataset = LatentTrajectoryDataset(
        cfg.data.root_dir,
        latents_file_name=cfg.data.latents_file_name,
        conditioning_file_name=cfg.data.conditioning_file_name,
        targets_dir_name=cfg.data.targets_dir_name,
        target_eps_file_name=cfg.data.target_eps_file_name,
        require_training_cache=cfg.data.require_training_cache,
    )
    val_dataset = LatentTrajectoryDataset(
        cfg.data.val_root_dir,
        latents_file_name=cfg.data.latents_file_name,
        conditioning_file_name=cfg.data.conditioning_file_name,
        targets_dir_name=cfg.data.targets_dir_name,
        target_eps_file_name=cfg.data.target_eps_file_name,
        require_training_cache=cfg.data.require_training_cache,
    )
    logger.info("Loaded {:,} train transitions", len(train_dataset))
    logger.info("Loaded {:,} val transitions", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
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

    trainer.train(train_loader, val_loader, max_train_steps=cfg.max_train_steps)


if __name__ == "__main__":
    main()
