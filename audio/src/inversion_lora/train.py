# ABOUTME: Train a LoRA on AudioLDM2 so that DDIM inversion becomes near-exact, by distilling
# ABOUTME: the frozen teacher's epsilon at x_t into the student's prediction at the cleaner x_t-1.

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import load_model  # noqa: E402

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_trajectory_batch,
    split_sample_ids,
)

FROZEN_COMPONENTS = (
    "vae",
    "text_encoder",
    "text_encoder_2",
    "language_model",
    "projection_model",
    "vocoder",
)


def git_sha() -> str:
    """Return the current commit SHA so checkpoints are traceable to their code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
    ).strip()


class AudioLDM2InversionTrainer:
    """Distils the frozen teacher's shifted epsilon into a LoRA on the AudioLDM2 UNet.

    The targets are precomputed, so the teacher is never run here; the LoRA stays enabled for
    the whole of training and the frozen weights receive no gradient.
    """

    def __init__(self, ldm, cfg: DictConfig, tracker: Any):
        self.ldm = ldm
        self.cfg = cfg
        self.tracker = tracker
        self.unet = ldm.model.unet
        self.device = ldm.device
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.global_step = 0
        self.baseline_reference = float("nan")

        self._freeze_components()
        inject_adapter_in_model(
            LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True)),
            self.unet,
            adapter_name=str(cfg.adapter_name),
        )
        for name, param in self.unet.named_parameters():
            param.requires_grad_("lora" in name.lower())
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        self.trainable_parameters = [p for p in self.unet.parameters() if p.requires_grad]
        if not self.trainable_parameters:
            raise RuntimeError("No LoRA parameters were marked trainable.")
        logger.info(
            "Trainable LoRA parameters: {:,} across {} tensors",
            sum(p.numel() for p in self.trainable_parameters),
            len(self.trainable_parameters),
        )

        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

    def _freeze_components(self) -> None:
        for name in FROZEN_COMPONENTS:
            component = getattr(self.ldm.model, name, None)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
                component.eval()

    def predict_noise(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the student UNet on the cleaner latent with the cached conditioning."""
        x_clean = batch["x_clean"].to(device=self.device, dtype=self.unet.dtype)
        timestep = batch["timestep"].to(device=self.device)
        hidden = batch["generated_prompt_embeds"].to(device=self.device, dtype=self.unet.dtype)
        t5_embeds = batch["t5_prompt_embeds"].to(device=self.device, dtype=self.unet.dtype)
        t5_mask = batch["t5_attention_mask"].to(device=self.device)

        assert x_clean.shape[0] == timestep.shape[0] == hidden.shape[0], (
            x_clean.shape,
            timestep.shape,
            hidden.shape,
        )

        model_input = self.ldm.model.scheduler.scale_model_input(x_clean, timestep)
        return self.ldm.unet_forward(
            model_input,
            timestep=timestep,
            encoder_hidden_states=hidden,
            class_labels=t5_embeds,
            encoder_attention_mask=t5_mask,
        )[0].sample

    def forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """No-CFG loss: MSE between the student's epsilon and the cached teacher epsilon."""
        student_eps = self.predict_noise(batch)
        target_eps = batch["target_eps"].to(device=self.device, dtype=self.unet.dtype)
        assert student_eps.shape == target_eps.shape, (student_eps.shape, target_eps.shape)
        return F.mse_loss(student_eps.float(), target_eps.float())

    @torch.no_grad()
    def baseline_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Loss with the LoRA disabled, i.e. the shift gap the LoRA has to close.

        Logged once at startup so the training curve has a meaningful reference: anything at or
        above this number means the adapter is not helping.
        """
        self.set_lora_enabled(False)
        try:
            return self.forward_loss(batch)
        finally:
            self.set_lora_enabled(True)

    def set_lora_enabled(self, enabled: bool) -> None:
        """Toggle every injected adapter; disabling restores the frozen teacher exactly."""
        toggled = 0
        for module in self.unet.modules():
            if module is not self.unet and hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled)
                toggled += 1
        if not toggled:
            raise AttributeError("No injected LoRA layers exposed enable_adapters().")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int | None) -> dict[str, float]:
        """Average validation loss over up to `max_batches` batches."""
        self.unet.eval()
        losses: list[float] = []
        for idx, batch in enumerate(tqdm(val_loader, desc="validation", leave=False)):
            if max_batches is not None and idx >= max_batches:
                break
            loss = self.forward_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss at step {self.global_step}")
            losses.append(float(loss))
        self.unet.train()
        return {"val/loss": sum(losses) / len(losses)} if losses else {}

    def save_checkpoint(self, filename: str, save_training_state: bool = False) -> Path:
        """Save the LoRA adapter, plus optimizer state when resuming matters."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = get_peft_model_state_dict(
            self.unet, adapter_name=str(self.cfg.adapter_name)
        )
        path = self.checkpoint_dir / filename
        torch.save(adapter_state, path)

        meta = {
            "global_step": self.global_step,
            "git_sha": git_sha(),
            "model_id": str(self.cfg.model_id),
            "num_inference_steps": int(self.cfg.num_inference_steps),
            "lora": OmegaConf.to_container(self.cfg.lora, resolve=True),
            "adapter_name": str(self.cfg.adapter_name),
        }
        with path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if save_training_state:
            torch.save(
                {
                    "global_step": self.global_step,
                    "lora_state_dict": adapter_state,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                self.checkpoint_dir / f"training_state_{filename}",
            )
        logger.info("Saved checkpoint {}", path)
        return path

    def load_training_state(self, path: str | Path) -> int:
        """Restore adapter and optimizer state, returning the step to resume from."""
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        set_peft_model_state_dict(
            self.unet, state["lora_state_dict"], adapter_name=str(self.cfg.adapter_name)
        )
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        return int(state["global_step"])

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        max_train_steps: int,
        initial_global_step: int = 0,
    ) -> None:
        """Run the training loop with gradient accumulation and a hard non-finite guard."""
        self.global_step = initial_global_step
        accum = max(1, int(self.cfg.gradient_accumulation_steps))
        micro_step = 0
        recent: list[float] = []
        self.unet.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=max_train_steps, initial=self.global_step, desc="train steps")

        logged_first = False
        while self.global_step < max_train_steps:
            for batch in train_loader:
                if not logged_first:
                    self.baseline_reference = float(self.baseline_loss(batch))
                    logger.info(
                        "First batch: x_clean={} t={} t5={} | LoRA-disabled loss={:.6f}",
                        tuple(batch["x_clean"].shape),
                        batch["timestep"].tolist(),
                        tuple(batch["t5_prompt_embeds"].shape),
                        self.baseline_reference,
                    )
                    self.tracker.log(
                        {"train/loss_lora_disabled": self.baseline_reference},
                        step=self.global_step,
                    )
                    logged_first = True

                loss = self.forward_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at step {self.global_step}: {float(loss)}"
                    )
                (loss / accum).backward()
                recent.append(float(loss))
                micro_step += 1
                if micro_step % accum:
                    continue

                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        float(self.cfg.max_grad_norm),
                        error_if_nonfinite=True,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                progress.update(1)

                if self._due(self.cfg.log_every_steps):
                    train_loss = sum(recent) / len(recent)
                    # Also to the logger: with the tracker disabled the curve would otherwise be
                    # invisible, which makes a smoke run unreadable.
                    logger.info(
                        "step {} train/loss={:.6f} (LoRA-disabled baseline {:.6f})",
                        self.global_step,
                        train_loss,
                        self.baseline_reference,
                    )
                    self.tracker.log({"train/loss": train_loss}, step=self.global_step)
                    recent = []
                if val_loader is not None and self._due(self.cfg.eval_every_steps):
                    metrics = self.validate(val_loader, self.cfg.max_val_batches)
                    if metrics:
                        logger.info("step {} {}", self.global_step, metrics)
                        self.tracker.log(metrics, step=self.global_step)
                if self._due(self.cfg.save_every_steps):
                    self.save_checkpoint(
                        f"checkpoint_step_{self.global_step}.pt",
                        save_training_state=bool(self.cfg.save_training_state),
                    )
                if self.global_step >= max_train_steps:
                    break

        progress.close()
        self.save_checkpoint("checkpoint_final.pt", save_training_state=bool(self.cfg.save_training_state))

    def _due(self, interval: Any) -> bool:
        interval = int(interval)
        return interval > 0 and self.global_step > 0 and self.global_step % interval == 0


class NullTracker:
    """Stand-in tracker so CPU smoke runs need no W&B."""

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


@hydra.main(config_path="../../config", config_name="train_inversion_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the AudioLDM2 inversion LoRA on cached trajectories."""
    load_dotenv(AUDIO_ROOT / ".env", override=False)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type != "cpu":
        torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))

    train_ids, val_ids = split_sample_ids(cfg.data_root, float(cfg.val_fraction), int(cfg.seed))
    train_dataset = AudioLDM2TrajectoryDataset(cfg.data_root, sample_ids=train_ids)
    logger.info(
        "train: {:,} transitions from {} trajectories", len(train_dataset), len(train_ids)
    )
    val_loader = None
    if val_ids:
        val_dataset = AudioLDM2TrajectoryDataset(cfg.data_root, sample_ids=val_ids)
        logger.info("val:   {:,} transitions from {} trajectories", len(val_dataset), len(val_ids))
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            collate_fn=collate_trajectory_batch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_trajectory_batch,
        drop_last=False,
    )

    ldm = load_model(str(cfg.model_id), device, int(cfg.num_inference_steps), edit_method="ddim")

    tracker: Any = NullTracker()
    if str(cfg.wandb_mode) != "disabled":
        import wandb

        tracker = wandb.init(
            project=str(cfg.wandb_project),
            name=cfg.run_name,
            mode=str(cfg.wandb_mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = AudioLDM2InversionTrainer(ldm, cfg, tracker)
    initial_step = 0
    if cfg.resume_from:
        initial_step = trainer.load_training_state(cfg.resume_from)
        logger.info("Resumed at step {}", initial_step)

    trainer.train(
        train_loader,
        val_loader,
        max_train_steps=int(cfg.max_train_steps),
        initial_global_step=initial_step,
    )
    tracker.finish()
    logger.success("Training finished at step {}", trainer.global_step)


if __name__ == "__main__":
    main()
