from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import wandb
from diffusers import StableDiffusionXLPipeline
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.utils import make_pipe


class SDXLInversionTrainer:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
        optimizer: Optimizer,
        lora_config: LoraConfig,
        tracker: Any,
        checkpoint_dir: Path | str,
        save_frequency: int = 100,
    ):
        self.pipe = pipe
        self.lora_config = lora_config
        self.model = pipe.unet
        inject_adapter_in_model(lora_config, self.model, adapter_name="inversion")
        self.optimizer = optimizer
        self.tracker = tracker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self.global_step = 0
        for epoch in tqdm(range(epochs)):
            logger.info("Epoch {}/{}", epoch + 1, epochs)
            self.tracker.log(self.train_epoch(train_loader), step=self.global_step)
            self.tracker.log(self.validation_epoch(val_loader), step=self.global_step)
            if epoch % self.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        self.tracker.finish()

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        losses = []

        for batch in tqdm(train_loader, desc="Training"):
            x_t = batch["x_t"].to(self.model.device)
            timestep = batch["timestep"].to(self.model.device)
            prompt_embeds = batch["prompt_embeds"].to(self.model.device)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.model.device)
            add_time_ids = batch["add_time_ids"].to(self.model.device)
            target_eps = batch["target_eps"].to(self.model.device)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            eps_pred = self.model(
                x_t,
                timestep,
                prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            loss = torch.nn.functional.mse_loss(eps_pred, target_eps)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())
            self.global_step += 1

        return {"Reconstruction Loss": sum(losses) / len(losses)}

    @torch.no_grad()
    def validation_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        for batch in tqdm(val_loader, desc="Validation"):
            eps_pred = self.model(
                batch["x_t"], batch["timestep"], batch["prompt_embeds"], batch["add_time_ids"]
            )
            loss = torch.nn.functional.mse_loss(eps_pred, batch["target_eps"])
            losses.append(loss.item())
        return {"Reconstruction Loss": sum(losses) / len(losses)}

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


def get_lora_config(lora_config: DictConfig) -> LoraConfig:
    return LoraConfig(**lora_config)


@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Training {cfg.model.model_id}")
    model_cfg = cfg.model
    lora_cfg = cfg.lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_cfg.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(model_cfg, device)
    lora_config = get_lora_config(lora_cfg)

    run = wandb.init(
        project="diff-inversion",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = SDXLInversionTrainer(
        pipe=pipe,
        optimizer=torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4),
        lora_config=lora_config,
        tracker=run,
        checkpoint_dir=cfg.checkpoint_dir,
        save_frequency=cfg.save_frequency,
    )

    train_dataset = LatentTrajectoryDataset(cfg.data.root_dir)
    val_dataset = LatentTrajectoryDataset(cfg.data.val_root_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    trainer.train(train_loader, val_loader, epochs=cfg.epochs)


if __name__ == "__main__":
    main()
