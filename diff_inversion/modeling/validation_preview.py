"""Validation preview rendering for SDXL inversion LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
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
from diff_inversion.eval.previews import channel_grid_image, fit_panel, write_noise_images
from diff_inversion.eval.sample_metrics import (
    image_pair_metrics,
    load_rgb_tensor,
    pair_metrics,
)
from diff_inversion.modeling.sdxl_sampling import (
    invert_latent_sdxl,
    reconstruct_latent_sdxl,
)


@dataclass
class PreviewResult:
    reconstruction: Image.Image
    reconstructed_image: Image.Image
    inverted_noise: torch.Tensor
    noise_predictions: list[tuple[int, Image.Image]]
    metrics: dict[str, float]


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
    wandb_reconstruction_images = []
    wandb_noise_images = []
    wandb_inverted_noise_images = []
    metric_rows: list[dict[str, float]] = []

    try:
        for sample_dir in sample_dirs:
            try:
                result = _render_validation_preview(
                    pipe=pipe,
                    cfg=cfg,
                    sample_dir=sample_dir,
                    original_scheduler=original_scheduler,
                )
            except Exception as exc:
                logger.warning("Validation preview failed for {}: {}", sample_dir, exc)
                continue

            output_path = preview_dir / f"{sample_dir.name}.png"
            reconstructed_path = preview_dir / f"{sample_dir.name}_reconstructed.png"
            result.reconstruction.save(output_path)
            result.reconstructed_image.save(reconstructed_path)
            logger.info("Saved validation preview: {}", output_path)

            image_metrics = image_pair_metrics(
                reference_path=sample_dir / "final.png",
                candidate_path=reconstructed_path,
                plain_threshold=float(cfg.validation_preview.plain_threshold),
            )
            if bool(cfg.validation_preview.calculate_lpips):
                image_metrics["lpips"] = _lpips_distance(
                    sample_dir / "final.png",
                    reconstructed_path,
                    device=pipe.device,
                )
            for key, value in image_metrics.items():
                if isinstance(value, bool) or value is None:
                    continue
                result.metrics[f"reconstruction/{key}"] = float(value)
            metric_rows.append(result.metrics)

            for inversion_step, noise_image in result.noise_predictions:
                noise_path = preview_dir / (
                    f"{sample_dir.name}_noise_pred_step_{inversion_step:03d}.png"
                )
                noise_image.save(noise_path)

            if bool(cfg.validation_preview.log_inverted_noise):
                inverted_noise_path = preview_dir / f"{sample_dir.name}_inverted_noise.png"
                write_noise_images(
                    inverted_noise_path,
                    initial_noise=_load_initial_noise(cfg, sample_dir),
                    inverted_noise=_squeeze_saved_batch(result.inverted_noise),
                    final_image_path=sample_dir / "final.png",
                )

            if tracker is not None:
                wandb_image = _make_wandb_image(
                    result.reconstruction,
                    caption=f"epoch {epoch + 1}: {sample_dir.name}",
                )
                if wandb_image is not None:
                    wandb_reconstruction_images.append(wandb_image)

                for inversion_step, noise_image in result.noise_predictions:
                    wandb_noise_image = _make_wandb_image(
                        noise_image,
                        caption=(
                            f"epoch {epoch + 1}: {sample_dir.name}, "
                            f"inversion_step={inversion_step}"
                        ),
                    )
                    if wandb_noise_image is not None:
                        wandb_noise_images.append(wandb_noise_image)

                if bool(cfg.validation_preview.log_inverted_noise):
                    inverted_image = _make_inverted_noise_preview(
                        cfg=cfg,
                        sample_dir=sample_dir,
                        inverted_noise=result.inverted_noise,
                    )
                    wandb_inverted_noise_image = _make_wandb_image(
                        inverted_image,
                        caption=f"epoch {epoch + 1}: {sample_dir.name}, inverted noise",
                    )
                    if wandb_inverted_noise_image is not None:
                        wandb_inverted_noise_images.append(wandb_inverted_noise_image)
    finally:
        pipe.scheduler = original_scheduler
        _set_lora_enabled(pipe, True)
        pipe.unet.train(was_training)

    scalar_payload = _mean_metrics(metric_rows)
    log_payload: dict[str, Any] = dict(scalar_payload)
    if tracker is not None:
        if wandb_reconstruction_images:
            log_payload["val/preview"] = wandb_reconstruction_images
        if wandb_noise_images:
            log_payload["val/noise_prediction_preview"] = wandb_noise_images
        if wandb_inverted_noise_images:
            log_payload["val/inverted_noise_preview"] = wandb_inverted_noise_images

    if log_payload:
        logger.info("Validation preview metrics: {}", scalar_payload)
        if tracker is not None:
            tracker.log(log_payload, step=global_step)


def _render_validation_preview(
    *,
    pipe,
    cfg: DictConfig,
    sample_dir: Path,
    original_scheduler,
) -> PreviewResult:
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
    noise_predictions = []
    noise_metrics = {}
    if bool(cfg.validation_preview.log_predicted_noise):
        noise_predictions, noise_metrics = _predict_noise_previews(
            pipe=pipe,
            cfg=cfg,
            sample_dir=sample_dir,
            prompt=prompt,
        )

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
    reconstruction = _make_preview_grid(final_image, reconstructed_image, abs_error)

    metrics = {
        **noise_metrics,
        **_inverted_noise_metrics(cfg, sample_dir, inverted_noise),
    }
    return PreviewResult(
        reconstruction=reconstruction,
        reconstructed_image=reconstructed_image,
        inverted_noise=inverted_noise.detach().cpu(),
        noise_predictions=noise_predictions,
        metrics=metrics,
    )


@torch.no_grad()
def _predict_noise_previews(
    *,
    pipe,
    cfg: DictConfig,
    sample_dir: Path,
    prompt: str,
) -> tuple[list[tuple[int, Image.Image]], dict[str, float]]:
    latent_paths = sorted((sample_dir / str(cfg.data.latents_dir_name)).glob("x_*.pt"))
    pred_noise_paths = sorted((sample_dir / str(cfg.data.pred_noises_dir_name)).glob("noise_*.pt"))
    timesteps = _read_json(sample_dir / "timesteps.json")
    if len(latent_paths) < 2 or len(pred_noise_paths) != len(latent_paths) - 1:
        return [], {}

    cond = encode_prompt_sdxl(
        pipe=pipe,
        prompt=prompt,
        negative_prompt="",
        height=int(cfg.model.height),
        width=int(cfg.model.width),
        do_classifier_free_guidance=False,
    )
    prompt_embeds = cond["prompt_embeds"].to(device=pipe.device, dtype=pipe.unet.dtype)
    pooled_prompt_embeds = cond["pooled_prompt_embeds"].to(
        device=pipe.device,
        dtype=pipe.unet.dtype,
    )
    add_time_ids = cond["add_time_ids"].to(device=pipe.device, dtype=pipe.unet.dtype)

    images: list[tuple[int, Image.Image]] = []
    rows: list[dict[str, float]] = []
    num_steps = len(pred_noise_paths)
    for inversion_step in list(cfg.validation_preview.noise_inversion_steps):
        inversion_step = int(inversion_step)
        if inversion_step < 0 or inversion_step >= num_steps:
            continue

        step_idx = num_steps - 1 - inversion_step
        input_latent = torch.load(
            latent_paths[step_idx + 1],
            map_location="cpu",
            weights_only=True,
        ).to(
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )
        input_latent = _squeeze_saved_batch(input_latent).unsqueeze(0)
        target_eps = _squeeze_saved_batch(
            torch.load(pred_noise_paths[step_idx], map_location="cpu", weights_only=True)
        ).float()
        timestep = torch.tensor([int(timesteps[step_idx])], device=pipe.device)

        pred_eps = pipe.unet(
            input_latent,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
            return_dict=False,
        )[0]
        pred_eps = _squeeze_saved_batch(pred_eps.detach().cpu())
        image = _make_noise_prediction_grid(target_eps, pred_eps)
        images.append((inversion_step, image))

        row = {
            f"noise_pred/inversion_step_{inversion_step:03d}/{key}": value
            for key, value in pair_metrics(target_eps.float(), pred_eps.float()).items()
        }
        rows.append(row)

    return images, _average_metric_rows(rows)


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
    latent = torch.load(latent_paths[-1], map_location="cpu", weights_only=True)
    return latent.to(device=pipe.device, dtype=pipe.unet.dtype)


def _read_prompt(sample_dir: Path) -> str:
    with (sample_dir / "prompt.json").open("r", encoding="utf-8") as f:
        return str(json.load(f)["prompt"])


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _squeeze_saved_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _load_initial_noise(cfg: DictConfig, sample_dir: Path) -> torch.Tensor:
    return _squeeze_saved_batch(
        torch.load(
            sample_dir / str(cfg.data.latents_dir_name) / "x_000.pt",
            map_location="cpu",
            weights_only=True,
        )
    ).float()


def _inverted_noise_metrics(
    cfg: DictConfig,
    sample_dir: Path,
    inverted_noise: torch.Tensor,
) -> dict[str, float]:
    initial_noise = _load_initial_noise(cfg, sample_dir)
    inverted_noise = _squeeze_saved_batch(inverted_noise.detach().cpu()).float()

    return {
        f"inverted_noise/{key}": value
        for key, value in pair_metrics(initial_noise, inverted_noise).items()
    }


def _make_noise_prediction_grid(target_eps: torch.Tensor, pred_eps: torch.Tensor) -> Image.Image:
    target_eps = _squeeze_saved_batch(target_eps).float().cpu()
    pred_eps = _squeeze_saved_batch(pred_eps).float().cpu()
    error = (pred_eps - target_eps).abs()

    combined = torch.cat([target_eps.flatten(), pred_eps.flatten()])
    vmin = float(combined.quantile(0.01).item())
    vmax = float(combined.quantile(0.99).item())
    error_vmax = float(error.quantile(0.99).item())

    panels = [
        ("target eps", channel_grid_image(target_eps, vmin, vmax)),
        ("pred eps", channel_grid_image(pred_eps, vmin, vmax)),
        ("abs error", channel_grid_image(error, 0.0, max(error_vmax, 1e-8))),
    ]
    return _make_labeled_grid(panels)


def _make_inverted_noise_preview(
    cfg: DictConfig,
    sample_dir: Path,
    inverted_noise: torch.Tensor,
) -> Image.Image:
    initial_noise = _load_initial_noise(cfg, sample_dir)
    inverted_noise = _squeeze_saved_batch(inverted_noise.detach().cpu()).float()
    error = (inverted_noise - initial_noise).abs()
    combined = torch.cat([initial_noise.flatten(), inverted_noise.flatten()])
    vmin = float(combined.quantile(0.01).item())
    vmax = float(combined.quantile(0.99).item())
    error_vmax = float(error.quantile(0.99).item())

    input_image = channel_grid_image(initial_noise, vmin, vmax)
    panels = []
    final_image_path = sample_dir / "final.png"
    if final_image_path.exists():
        with Image.open(final_image_path) as image:
            panels.append(("final", fit_panel(image.convert("RGB"), input_image.size)))
    panels.extend(
        [
            ("initial noise", input_image),
            ("inverted noise", channel_grid_image(inverted_noise, vmin, vmax)),
            ("abs error", channel_grid_image(error, 0.0, max(error_vmax, 1e-8))),
        ]
    )
    return _make_labeled_grid(panels)


def _make_labeled_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    label_height = 24
    panel_width, panel_height = panels[0][1].size
    grid = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for idx, (label, image) in enumerate(panels):
        x = idx * panel_width
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        grid.paste(image.convert("RGB"), (x, label_height))
    return grid


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


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {f"val_preview/{key}": value for key, value in _average_metric_rows(rows).items()}


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or value is None:
                continue
            numeric_value = float(value)
            if torch.isfinite(torch.tensor(numeric_value)).item():
                grouped.setdefault(key, []).append(numeric_value)
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def _lpips_distance(reference_path: Path, candidate_path: Path, device) -> float | None:
    try:
        from diff_inversion.eval.diversity import lpips_calculate_distances
    except ModuleNotFoundError as exc:
        logger.warning("LPIPS is not available; skipping validation preview LPIPS: {}", exc)
        return None

    try:
        with Image.open(reference_path) as reference_image:
            reference_size = reference_image.size

        reference = load_rgb_tensor(reference_path).unsqueeze(0)
        candidate = load_rgb_tensor(candidate_path, size=reference_size).unsqueeze(0)
        distance = lpips_calculate_distances(reference, candidate, device=torch.device(device))
    except ModuleNotFoundError as exc:
        logger.warning("LPIPS is not available; skipping validation preview LPIPS: {}", exc)
        return None
    except Exception as exc:
        logger.warning("LPIPS disabled for validation preview: {}", exc)
        return None

    return float(distance.squeeze().item())


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()
