# ABOUTME: Train a LoRA on Stable Audio Open so that DPMSolver inversion becomes near-exact, by
# ABOUTME: distilling the frozen teacher's prediction at x_t into the student's at the cleaner x_t-1.

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_stable_audio_batch,
    split_sample_ids,
    transitions_below_timestep,
)
from src.inversion_lora.stable_audio import load_teacher  # noqa: E402
from src.inversion_lora.train import (  # noqa: E402
    AudioLDM2InversionTrainer,
    NullTracker,
)

STABLE_AUDIO_CONDITIONING_KEYS = ("text_audio",)


class StableAudioInversionTrainer(AudioLDM2InversionTrainer):
    """Distils the frozen teacher's shifted prediction into a LoRA on the Stable Audio DiT.

    Everything the base trainer does is model-independent -- adapter injection, EMA, the loss
    bands, checkpointing, the loop -- so only the forward differs: Stable Audio conditions on one
    cross-attention tensor plus duration and rotary embeddings held by the teacher, and its
    denoiser takes per-example timesteps directly.
    """

    def log_first_batch(self, batch: dict[str, Any]) -> None:
        """Print the first training batch's shapes, Stable Audio's single conditioning included."""
        logger.info(
            "First batch: x_clean={} t={} text_audio={}",
            tuple(batch["x_clean"].shape),
            batch["timestep"].tolist(),
            tuple(batch["text_audio"].shape),
        )

    def predict_noise(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the student DiT on the cleaner latent with the cached conditioning."""
        x_clean = batch["x_clean"].to(device=self.device, dtype=self.unet.dtype)
        timestep = batch["timestep"].to(device=self.device)
        text_audio = batch["text_audio"].to(device=self.device, dtype=self.unet.dtype)
        assert x_clean.shape[0] == timestep.shape[0] == text_audio.shape[0], (
            x_clean.shape,
            timestep.shape,
            text_audio.shape,
        )
        # DPMSolverMultistep does no input preconditioning, so scale_model_input is the identity
        # here and the raw latent is what the teacher saw. It is not the identity for the native
        # cosine scheduler; see output/sao_probe/REPORT.md.
        return self.ldm.forward(x_clean, timestep, text_audio)


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None, set[int]]:
    """Build the train and validation loaders over a cached Stable Audio trajectory set.

    Splits by trajectory, optionally restricts training to the cleanest tail of the schedule, and
    scores validation on a fixed random subset so the reported loss covers the whole schedule
    rather than the first few trajectories' noisiest steps.

    Args:
        cfg: The resolved training config.

    Returns:
        `(train_loader, val_loader, val_ids)`; `val_loader` is None when nothing is held out.
    """
    train_ids, val_ids = split_sample_ids(cfg.data_root, float(cfg.val_fraction), int(cfg.seed))
    train_dataset = AudioLDM2TrajectoryDataset(
        cfg.data_root, sample_ids=train_ids, conditioning_keys=STABLE_AUDIO_CONDITIONING_KEYS
    )
    logger.info("train: {:,} transitions from {} trajectories", len(train_dataset), len(train_ids))
    if cfg.train_max_timestep:
        keep = transitions_below_timestep(train_dataset, int(cfg.train_max_timestep))
        logger.info(
            "train: restricted to t <= {}: {:,} of {:,} transitions",
            int(cfg.train_max_timestep),
            len(keep),
            len(train_dataset),
        )
        train_dataset = Subset(train_dataset, keep)

    val_loader = None
    if val_ids:
        val_dataset = AudioLDM2TrajectoryDataset(
            cfg.data_root, sample_ids=val_ids, conditioning_keys=STABLE_AUDIO_CONDITIONING_KEYS
        )
        logger.info("val:   {:,} transitions from {} trajectories", len(val_dataset), len(val_ids))
        capped = int(cfg.max_val_batches) * int(cfg.batch_size)
        if cfg.max_val_batches and capped < len(val_dataset):
            order = torch.randperm(
                len(val_dataset), generator=torch.Generator().manual_seed(int(cfg.seed))
            )
            val_dataset = Subset(val_dataset, order[:capped].tolist())
            logger.info("val:   scoring a fixed random subset of {:,} transitions", capped)
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            collate_fn=collate_stable_audio_batch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_stable_audio_batch,
        drop_last=False,
    )
    return train_loader, val_loader, val_ids


@hydra.main(
    config_path="../../config", config_name="train_inversion_lora_stable_audio", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Train the Stable Audio Open inversion LoRA on cached trajectories."""
    # override=True matches env.py: .env is the only place configuration is edited, so a stale
    # exported variable in the submitting shell must not win.
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type != "cpu":
        torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))

    train_loader, val_loader, _ = build_loaders(cfg)

    teacher = load_teacher(
        str(cfg.model_id),
        device,
        int(cfg.num_inference_steps),
        duration_s=cfg.duration_s,
        schedule=str(cfg.schedule),
    )

    tracker: Any = NullTracker()
    if str(cfg.wandb_mode) != "disabled":
        import wandb

        tracker = wandb.init(
            project=str(cfg.wandb_project),
            name=cfg.run_name,
            mode=str(cfg.wandb_mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = StableAudioInversionTrainer(teacher, cfg, tracker)
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
