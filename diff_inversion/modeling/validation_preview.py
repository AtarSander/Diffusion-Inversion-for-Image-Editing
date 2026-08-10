"""Optional validation previews for SDXL inversion LoRA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from diff_inversion.data.generate_sdxl_samples import (
    decode_latent_to_pil,
    encode_prompt_sdxl,
    has_sdxl_conditioning,
)
from diff_inversion.eval.lora import set_unet_lora_enabled
from diff_inversion.eval.previews import (
    channel_grid_image,
    fit_panel,
    pca_rgb_image,
    write_image_comparison,
    write_noise_images,
)
from diff_inversion.eval.sample_metrics import image_pair_metrics, load_rgb_tensor, pair_metrics
from diff_inversion.modeling.sdxl_sampling import (
    invert_latent_sdxl,
    reconstruct_latent_sdxl,
)

CFG_TARGET_MODES = {"cfg"}
CFG_SINGLE_PASS_TARGET_MODES = {"cfg_single_pass"}
BRANCH_TARGET_MODES = {
    "unconditional",
    *CFG_TARGET_MODES,
    *CFG_SINGLE_PASS_TARGET_MODES,
}


def should_run_validation_preview(cfg: DictConfig, global_step: int) -> bool:
    if not bool(OmegaConf.select(cfg, "validation_preview.enabled", default=False)):
        return False
    every_steps = int(OmegaConf.select(cfg, "validation_preview.every_steps", default=0))
    return every_steps > 0 and global_step > 0 and global_step % every_steps == 0


@torch.no_grad()
def log_validation_preview(
    *,
    pipe,
    cfg: DictConfig,
    tracker: Any | None,
    checkpoint_dir: Path,
    global_step: int,
) -> None:
    sample_dirs = _sample_dirs(cfg)
    if not sample_dirs:
        logger.warning("Validation preview skipped: no validation sample directories found")
        return

    preview_dir = _preview_dir(cfg, checkpoint_dir, global_step)
    preview_dir.mkdir(parents=True, exist_ok=True)

    original_scheduler = pipe.scheduler
    was_training = pipe.unet.training
    pipe.unet.eval()

    metric_rows: list[dict[str, float]] = []
    reconstruction_images = []
    noise_prediction_images = []
    inverted_noise_images = []

    try:
        for sample_dir in sample_dirs:
            try:
                row, images = _render_sample_preview(
                    pipe=pipe,
                    cfg=cfg,
                    sample_dir=sample_dir,
                    preview_dir=preview_dir,
                    original_scheduler=original_scheduler,
                )
            except Exception as exc:
                logger.warning("Validation preview failed for {}: {}", sample_dir, exc)
                continue

            metric_rows.append(row)
            if tracker is not None:
                _append_wandb_images(
                    reconstruction_images,
                    images["reconstruction"],
                    f"step {global_step}: {sample_dir.name}",
                )
                _append_wandb_images(
                    noise_prediction_images,
                    images["noise_prediction"],
                    f"step {global_step}: {sample_dir.name}, noise prediction",
                )
                _append_wandb_images(
                    inverted_noise_images,
                    images["inverted_noise"],
                    f"step {global_step}: {sample_dir.name}, inverted noise",
                )
    finally:
        pipe.scheduler = original_scheduler
        set_unet_lora_enabled(pipe, True)
        pipe.unet.train(was_training)

    payload: dict[str, Any] = _mean_metrics(metric_rows)
    if reconstruction_images:
        payload["val_preview/reconstruction"] = reconstruction_images
    if noise_prediction_images:
        payload["val_preview/noise_prediction"] = noise_prediction_images
    if inverted_noise_images:
        payload["val_preview/inverted_noise"] = inverted_noise_images

    if payload and tracker is not None:
        tracker.log(payload, step=global_step)
    if metric_rows:
        logger.info("Validation preview metrics: {}", _mean_metrics(metric_rows))


def _render_sample_preview(
    *,
    pipe,
    cfg: DictConfig,
    sample_dir: Path,
    preview_dir: Path,
    original_scheduler,
) -> tuple[dict[str, float], dict[str, list[Path]]]:
    trajectory = _load_trajectory(cfg, sample_dir)
    target_eps = _load_target_eps(cfg, sample_dir)
    timesteps = _read_json(sample_dir / "timesteps.json")
    final_image_path = sample_dir / str(
        OmegaConf.select(cfg, "validation_preview.final_image_name", default="final.png")
    )
    prompt = _read_prompt(sample_dir)
    target_mode = _training_target_mode(cfg)
    guidance_scale = _preview_guidance_scale(cfg, sample_dir)
    cond = encode_prompt_sdxl(
        pipe=pipe,
        prompt=prompt,
        negative_prompt="",
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )

    images: dict[str, list[Path]] = {
        "reconstruction": [],
        "noise_prediction": [],
        "inverted_noise": [],
    }
    metrics: dict[str, float] = {}

    set_unet_lora_enabled(pipe, True)
    if bool(OmegaConf.select(cfg, "validation_preview.log_predicted_noise", default=True)):
        noise_images, noise_metrics = _write_noise_prediction_previews(
            pipe=pipe,
            cfg=cfg,
            sample_dir=sample_dir,
            preview_dir=preview_dir,
            trajectory=trajectory,
            target_eps=target_eps,
            timesteps=timesteps,
        )
        images["noise_prediction"].extend(noise_images)
        metrics.update(noise_metrics)

    inverted_noise = invert_latent_sdxl(
        pipe=pipe,
        final_latent=_squeeze_batch(trajectory[-1]).unsqueeze(0),
        cond=cond,
        scheduler_config=original_scheduler.config,
        num_inference_steps=int(cfg.model.num_inference_steps),
        guidance_scale=guidance_scale,
        single_conditional_prediction=target_mode in CFG_SINGLE_PASS_TARGET_MODES,
    )
    metrics.update(
        {
            f"inverted_noise/{key}": value
            for key, value in pair_metrics(
                _squeeze_batch(trajectory[0]).float(),
                _squeeze_batch(inverted_noise.detach().cpu()).float(),
            ).items()
        }
    )

    if bool(OmegaConf.select(cfg, "validation_preview.log_inverted_noise", default=True)):
        inverted_preview = preview_dir / f"{sample_dir.name}_inverted_noise.png"
        noise_paths = write_noise_images(
            inverted_preview,
            initial_noise=_squeeze_batch(trajectory[0]).float(),
            inverted_noise=_squeeze_batch(inverted_noise.detach().cpu()).float(),
            final_image_path=final_image_path,
        )
        images["inverted_noise"].append(inverted_preview)
        for key in (
            "pca_input_noise_image_path",
            "pca_inverted_noise_image_path",
            "pca_abs_error_image_path",
        ):
            if key in noise_paths:
                images["inverted_noise"].append(Path(noise_paths[key]))

    pipe.scheduler = original_scheduler
    use_lora_for_reconstruction = bool(
        OmegaConf.select(
            cfg,
            "validation_preview.use_lora_for_reconstruction",
            default=False,
        )
    )
    set_unet_lora_enabled(pipe, use_lora_for_reconstruction)
    reconstructed_latent, _ = reconstruct_latent_sdxl(
        pipe=pipe,
        noise_latent=inverted_noise,
        cond=cond,
        num_inference_steps=int(cfg.model.num_inference_steps),
        guidance_scale=guidance_scale,
        single_conditional_prediction=(
            use_lora_for_reconstruction and target_mode in CFG_SINGLE_PASS_TARGET_MODES
        ),
    )
    reconstructed_image = decode_latent_to_pil(
        pipe,
        reconstructed_latent.to(device=pipe.device),
    ).convert("RGB")

    reconstructed_path = preview_dir / f"{sample_dir.name}_reconstructed.png"
    reconstructed_image.save(reconstructed_path)

    if final_image_path.exists():
        comparison_path = preview_dir / f"{sample_dir.name}_reconstruction.png"
        write_image_comparison(
            comparison_path,
            final_image_path=final_image_path,
            reconstructed_image_path=reconstructed_path,
            plain_threshold=float(
                OmegaConf.select(cfg, "validation_preview.plain_threshold", default=0.03)
            ),
        )
        images["reconstruction"].append(comparison_path)
        image_metrics = image_pair_metrics(
            reference_path=final_image_path,
            candidate_path=reconstructed_path,
            plain_threshold=float(
                OmegaConf.select(cfg, "validation_preview.plain_threshold", default=0.03)
            ),
        )
        if bool(OmegaConf.select(cfg, "validation_preview.calculate_lpips", default=False)):
            image_metrics["lpips"] = _lpips_distance(
                reference_path=final_image_path,
                candidate_path=reconstructed_path,
                device=pipe.device,
            )
        metrics.update(_prefix_numeric("reconstruction", image_metrics))

    return metrics, images


@torch.no_grad()
def _write_noise_prediction_previews(
    *,
    pipe,
    cfg: DictConfig,
    sample_dir: Path,
    preview_dir: Path,
    trajectory: torch.Tensor,
    target_eps: torch.Tensor,
    timesteps: list[int],
) -> tuple[list[Path], dict[str, float]]:
    cond = torch.load(_conditioning_path(cfg, sample_dir), map_location="cpu")
    target_mode = _training_target_mode(cfg)
    guidance_scale = _preview_guidance_scale(cfg, sample_dir)
    target_eps_uncond = None
    if target_mode in BRANCH_TARGET_MODES:
        target_eps_uncond = _load_target_eps_uncond(cfg, sample_dir)

    prompt_embeds = _ensure_batch(cond["prompt_embeds"]).to(
        device=pipe.device,
        dtype=pipe.unet.dtype,
    )
    negative_prompt_embeds = None
    if target_mode in BRANCH_TARGET_MODES:
        if "negative_prompt_embeds" not in cond:
            raise KeyError(
                f"Missing negative_prompt_embeds in {_conditioning_path(cfg, sample_dir)}"
            )
        negative_prompt_embeds = _ensure_batch(cond["negative_prompt_embeds"]).to(
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )

    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    add_time_ids = None
    if has_sdxl_conditioning(cond):
        pooled_prompt_embeds = _ensure_batch(cond["pooled_prompt_embeds"]).to(
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )
        if target_mode in BRANCH_TARGET_MODES:
            if "negative_pooled_prompt_embeds" not in cond:
                raise KeyError(
                    f"Missing negative_pooled_prompt_embeds in {_conditioning_path(cfg, sample_dir)}"
                )
            negative_pooled_prompt_embeds = _ensure_batch(
                cond["negative_pooled_prompt_embeds"]
            ).to(
                device=pipe.device,
                dtype=pipe.unet.dtype,
            )
        add_time_ids = _ensure_batch(cond["add_time_ids"]).to(
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )

    num_steps = int(target_eps.shape[0])
    preview_paths: list[Path] = []
    metric_rows: list[dict[str, float]] = []
    for inversion_step in list(
        OmegaConf.select(cfg, "validation_preview.noise_inversion_steps", default=[0, 10, 25, 49])
    ):
        inversion_step = int(inversion_step)
        if inversion_step < 0 or inversion_step >= num_steps:
            continue

        step_idx = num_steps - 1 - inversion_step
        timestep = _transition_timestep(timesteps, step_idx, int(trajectory.shape[0]))
        latent = (
            _squeeze_batch(trajectory[step_idx + 1])
            .unsqueeze(0)
            .to(
                device=pipe.device,
                dtype=pipe.unet.dtype,
            )
        )
        scheduler_timestep = torch.tensor([timestep], device=pipe.device)
        if target_mode in CFG_TARGET_MODES:
            model_latent = torch.cat([latent, latent], dim=0)
            model_timestep = scheduler_timestep.repeat(2)
            model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        elif target_mode == "unconditional":
            model_latent = latent
            model_timestep = scheduler_timestep
            model_prompt_embeds = negative_prompt_embeds
        else:
            model_latent = latent
            model_timestep = scheduler_timestep
            model_prompt_embeds = prompt_embeds

        model_input = pipe.scheduler.scale_model_input(model_latent, model_timestep)
        unet_kwargs = {}
        if pooled_prompt_embeds is not None and add_time_ids is not None:
            if target_mode in CFG_TARGET_MODES:
                text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                    dim=0,
                )
                time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            elif target_mode == "unconditional":
                text_embeds = negative_pooled_prompt_embeds
                time_ids = add_time_ids
            else:
                text_embeds = pooled_prompt_embeds
                time_ids = add_time_ids
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
        pred_eps = pipe.unet(
            model_input,
            model_timestep,
            encoder_hidden_states=model_prompt_embeds,
            return_dict=False,
            **unet_kwargs,
        )[0]

        target_cond = _squeeze_batch(target_eps[step_idx]).float()
        if target_mode in CFG_TARGET_MODES:
            pred_uncond, pred_cond = pred_eps.detach().cpu().float().chunk(2)
            target_uncond = _squeeze_batch(target_eps_uncond[step_idx]).float()
            target = target_uncond + guidance_scale * (target_cond - target_uncond)
            predicted = _squeeze_batch(
                pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            ).float()
        elif target_mode in CFG_SINGLE_PASS_TARGET_MODES:
            target_uncond = _squeeze_batch(target_eps_uncond[step_idx]).float()
            target = target_uncond + guidance_scale * (target_cond - target_uncond)
            predicted = _squeeze_batch(pred_eps.detach().cpu()).float()
        elif target_mode == "unconditional":
            target = _squeeze_batch(target_eps_uncond[step_idx]).float()
            predicted = _squeeze_batch(pred_eps.detach().cpu()).float()
        else:
            target = target_cond
            predicted = _squeeze_batch(pred_eps.detach().cpu()).float()
        preview_path = preview_dir / (
            f"{sample_dir.name}_noise_pred_inv_step_{inversion_step:03d}.png"
        )
        _write_noise_prediction_grid(preview_path, target, predicted)
        preview_paths.append(preview_path)
        metric_rows.append(
            {
                f"noise_prediction/inversion_step_{inversion_step:03d}/{key}": value
                for key, value in pair_metrics(target, predicted).items()
            }
        )

    return preview_paths, _average_metrics(metric_rows)


def _write_noise_prediction_grid(
    output_path: Path,
    target_eps: torch.Tensor,
    pred_eps: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = (pred_eps - target_eps).abs()
    combined = torch.cat([target_eps.flatten(), pred_eps.flatten()])
    vmin = float(combined.quantile(0.01).item())
    vmax = float(combined.quantile(0.99).item())
    error_vmax = float(error.quantile(0.99).item())

    pca_target, components = pca_rgb_image(target_eps)
    pca_pred, _ = pca_rgb_image(pred_eps, components=components)
    pca_error, _ = pca_rgb_image(error)
    panels = [
        ("target eps", channel_grid_image(target_eps, vmin, vmax)),
        ("pred eps", channel_grid_image(pred_eps, vmin, vmax)),
        ("abs error", channel_grid_image(error, 0.0, max(error_vmax, 1e-8))),
        ("target eps PCA", pca_target),
        ("pred eps PCA", pca_pred),
        ("abs error PCA", pca_error),
    ]

    panel_size = panels[0][1].size
    label_height = 20
    canvas = Image.new(
        "RGB",
        (panel_size[0] * len(panels), panel_size[1] + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        x = idx * panel_size[0]
        draw.text((x + 4, 4), label, fill="black")
        canvas.paste(fit_panel(image, panel_size), (x, label_height))
    canvas.save(output_path)


def _sample_dirs(cfg: DictConfig) -> list[Path]:
    val_roots = OmegaConf.select(cfg, "data.val_root_dirs", default=None)
    if val_roots:
        roots = list(val_roots)
    else:
        val_root = OmegaConf.select(cfg, "data.val_root_dir", default=None)
        roots = [val_root] if val_root else []
    if not roots:
        return []
    limit = int(OmegaConf.select(cfg, "validation_preview.num_samples", default=2))
    if limit <= 0:
        return []
    samples_per_root = [
        sorted(path for path in _resolve_path(root).glob("sample_*") if path.is_dir())
        for root in roots
    ]
    selected: list[Path] = []
    max_samples = max((len(paths) for paths in samples_per_root), default=0)
    for sample_idx in range(max_samples):
        for paths in samples_per_root:
            if sample_idx < len(paths):
                selected.append(paths[sample_idx])
                if len(selected) >= limit:
                    return selected
    return selected


def _preview_dir(cfg: DictConfig, checkpoint_dir: Path, global_step: int) -> Path:
    output_dir = OmegaConf.select(cfg, "validation_preview.output_dir")
    if output_dir:
        return _resolve_path(output_dir) / f"step_{global_step:07d}"
    return checkpoint_dir / "previews" / f"step_{global_step:07d}"


def _load_trajectory(cfg: DictConfig, sample_dir: Path) -> torch.Tensor:
    latents_dir = sample_dir / str(
        OmegaConf.select(cfg, "data.latents_dir_name", default="latents")
    )
    trajectory_path = latents_dir / str(
        OmegaConf.select(cfg, "data.latents_file_name", default="trajectory.pt")
    )
    if trajectory_path.exists():
        return torch.load(trajectory_path, map_location="cpu")

    latent_paths = sorted(latents_dir.glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No latent trajectory found in {latents_dir}")
    return torch.stack([torch.load(path, map_location="cpu") for path in latent_paths], dim=0)


def _load_target_eps(cfg: DictConfig, sample_dir: Path) -> torch.Tensor:
    path = sample_dir / str(OmegaConf.select(cfg, "data.targets_dir_name", default="targets"))
    path = path / str(OmegaConf.select(cfg, "data.target_eps_file_name", default="target_eps.pt"))
    return torch.load(path, map_location="cpu")


def _load_target_eps_uncond(cfg: DictConfig, sample_dir: Path) -> torch.Tensor:
    path = sample_dir / str(OmegaConf.select(cfg, "data.targets_dir_name", default="targets"))
    path = path / str(
        OmegaConf.select(
            cfg,
            "data.target_uncond_eps_file_name",
            default="target_eps_uncond.pt",
        )
    )
    return torch.load(path, map_location="cpu")


def _conditioning_path(cfg: DictConfig, sample_dir: Path) -> Path:
    return sample_dir / str(
        OmegaConf.select(cfg, "data.conditioning_file_name", default="conditioning.pt")
    )


def _training_target_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "training_target.mode", default="conditional")).lower()


def _preview_guidance_scale(cfg: DictConfig, sample_dir: Path) -> float:
    value = OmegaConf.select(cfg, "training_target.guidance_scale", default=None)
    if value is None:
        value = _read_json(sample_dir / "meta.json").get(
            "guidance_scale",
            OmegaConf.select(cfg, "model.guidance_scale", default=1.0),
        )
    return float(value)


def _read_prompt(sample_dir: Path) -> str:
    with (sample_dir / "prompt.json").open("r", encoding="utf-8") as f:
        return str(json.load(f).get("prompt", ""))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _transition_timestep(timesteps: list[int], step_idx: int, trajectory_length: int) -> int:
    if len(timesteps) == trajectory_length:
        return int(timesteps[step_idx + 1])
    if len(timesteps) == trajectory_length - 1:
        return int(timesteps[step_idx])
    raise ValueError(
        "Unexpected timestep count for trajectory: "
        f"got {len(timesteps)}, expected {trajectory_length} or {trajectory_length - 1}"
    )


def _squeeze_batch(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.ndim >= 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        return tensor
    return tensor.unsqueeze(0)


def _prefix_numeric(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}/{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _lpips_distance(reference_path: Path, candidate_path: Path, device) -> float | None:
    try:
        from diff_inversion.eval.diversity import lpips_calculate_distances
    except ModuleNotFoundError as exc:
        logger.warning("LPIPS unavailable for validation preview: {}", exc)
        return None

    try:
        with Image.open(reference_path) as reference_image:
            reference_size = reference_image.size
        reference = load_rgb_tensor(reference_path).unsqueeze(0)
        candidate = load_rgb_tensor(candidate_path, size=reference_size).unsqueeze(0)
        distance = lpips_calculate_distances(reference, candidate, device=torch.device(device))
    except Exception as exc:
        logger.warning("LPIPS failed for validation preview: {}", exc)
        return None

    return float(distance.squeeze().item())


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {f"val_preview/{key}": value for key, value in _average_metrics(rows).items()}


def _average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            value_t = torch.tensor(float(value))
            if torch.isfinite(value_t).item():
                grouped.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def _wandb_image(path: Path, caption: str):
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    return wandb.Image(str(path), caption=caption)


def _append_wandb_images(target: list[Any], paths: list[Path], caption: str) -> None:
    for path in paths:
        image = _wandb_image(path, caption)
        if image is not None:
            target.append(image)


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()
