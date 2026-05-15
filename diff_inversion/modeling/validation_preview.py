"""Validation preview rendering for SDXL inversion LoRA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from PIL import Image, ImageChops, ImageDraw
import torch

from diff_inversion.data.generate_sdxl_samples import (
    decode_latent_to_pil,
    encode_prompt_sdxl,
)
from diff_inversion.modeling.sdxl_sampling import (
    invert_latent_sdxl,
    reconstruct_latent_sdxl,
)


def should_run_validation_preview(cfg: DictConfig, epoch: int) -> bool:
    if not bool(cfg.validation_preview.enabled):
        return False
    frequency = int(cfg.validation_preview.every_epochs)
    return frequency > 0 and (epoch + 1) % frequency == 0


@torch.no_grad()
def log_validation_preview(
    *,
    pipe,
    cfg: DictConfig,
    tracker: Any | None,
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    sample_dirs = _validation_preview_sample_dirs(cfg)
    if not sample_dirs:
        logger.warning("Validation preview skipped: no sample directories found")
        return

    preview_dir = _validation_preview_output_dir(cfg, checkpoint_dir, epoch)
    preview_dir.mkdir(parents=True, exist_ok=True)

    original_scheduler = pipe.scheduler
    was_training = pipe.unet.training
    pipe.unet.eval()
    wandb_images = []

    try:
        for sample_dir in sample_dirs:
            try:
                preview = _render_validation_preview(
                    pipe=pipe,
                    cfg=cfg,
                    sample_dir=sample_dir,
                    original_scheduler=original_scheduler,
                )
            except Exception as exc:
                logger.warning("Validation preview failed for {}: {}", sample_dir, exc)
                continue

            output_path = preview_dir / f"{sample_dir.name}.png"
            preview.save(output_path)
            logger.info("Saved validation preview: {}", output_path)

            if tracker is not None:
                wandb_image = _make_wandb_image(
                    preview,
                    caption=f"epoch {epoch + 1}: {sample_dir.name}",
                )
                if wandb_image is not None:
                    wandb_images.append(wandb_image)
    finally:
        pipe.scheduler = original_scheduler
        _set_lora_enabled(pipe, True)
        pipe.unet.train(was_training)

    if tracker is not None and wandb_images:
        tracker.log({"val/preview": wandb_images}, step=global_step)


def _render_validation_preview(
    *,
    pipe,
    cfg: DictConfig,
    sample_dir: Path,
    original_scheduler,
) -> Image.Image:
    prompt = _read_prompt(sample_dir)
    final_latent = _load_final_latent(pipe, cfg, sample_dir)
    final_image = Image.open(sample_dir / "final.png").convert("RGB")
    cond = encode_prompt_sdxl(
        pipe=pipe,
        prompt=prompt,
        negative_prompt="",
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )

    _set_lora_enabled(pipe, True)
    inverted_noise = invert_latent_sdxl(
        pipe=pipe,
        final_latent=final_latent,
        cond=cond,
        scheduler_config=original_scheduler.config,
        num_inference_steps=int(cfg.model.num_inference_steps),
        guidance_scale=float(cfg.model.guidance_scale),
    )

    pipe.scheduler = original_scheduler
    _set_lora_enabled(pipe, bool(cfg.validation_preview.use_lora_for_reconstruction))
    reconstructed_latent, _ = reconstruct_latent_sdxl(
        pipe=pipe,
        noise_latent=inverted_noise,
        cond=cond,
        num_inference_steps=int(cfg.model.num_inference_steps),
        guidance_scale=float(cfg.model.guidance_scale),
    )
    reconstructed_image = decode_latent_to_pil(
        pipe,
        reconstructed_latent.to(device=pipe.device),
    ).convert("RGB")

    if reconstructed_image.size != final_image.size:
        reconstructed_image = reconstructed_image.resize(final_image.size)
    abs_error = ImageChops.difference(final_image, reconstructed_image)
    return _make_preview_grid(final_image, reconstructed_image, abs_error)


def _validation_preview_sample_dirs(cfg: DictConfig) -> list[Path]:
    if not cfg.data.val_root_dir:
        return []
    val_root = _resolve_path(cfg.data.val_root_dir)
    sample_dirs = sorted(path for path in val_root.glob("sample_*") if path.is_dir())
    return sample_dirs[: int(cfg.validation_preview.num_samples)]


def _validation_preview_output_dir(
    cfg: DictConfig,
    checkpoint_dir: Path,
    epoch: int,
) -> Path:
    output_dir = cfg.validation_preview.output_dir
    if output_dir:
        return _resolve_path(output_dir) / f"epoch_{epoch + 1:03d}"
    return checkpoint_dir / "previews" / f"epoch_{epoch + 1:03d}"


def _load_final_latent(pipe, cfg: DictConfig, sample_dir: Path) -> torch.Tensor:
    latent_paths = sorted((sample_dir / str(cfg.data.latents_dir_name)).glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No latent files found in {sample_dir}")
    latent = torch.load(latent_paths[-1], map_location="cpu", weights_only=False)
    return latent.to(device=pipe.device, dtype=pipe.unet.dtype)


def _read_prompt(sample_dir: Path) -> str:
    with (sample_dir / "prompt.json").open("r", encoding="utf-8") as f:
        return str(json.load(f)["prompt"])


def _make_preview_grid(
    final_image: Image.Image,
    reconstructed_image: Image.Image,
    abs_error: Image.Image,
) -> Image.Image:
    width, height = final_image.size
    label_height = 28
    grid = Image.new("RGB", (width * 3, height + label_height), "white")
    draw = ImageDraw.Draw(grid)
    panels = [
        ("final", final_image),
        ("reconstructed", reconstructed_image),
        ("abs_error", abs_error),
    ]
    for idx, (label, image) in enumerate(panels):
        x = idx * width
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        grid.paste(image, (x, label_height))
    return grid


def _set_lora_enabled(pipe, enabled: bool) -> None:
    if enabled and hasattr(pipe.unet, "enable_adapters"):
        pipe.unet.enable_adapters()
    elif not enabled and hasattr(pipe.unet, "disable_adapters"):
        pipe.unet.disable_adapters()


def _make_wandb_image(image: Image.Image, caption: str):
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    return wandb.Image(image, caption=caption)


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()
