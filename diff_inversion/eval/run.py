"""Run evaluation over saved SDXL trajectory samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from diff_inversion.eval.clip_alignment import add_clip_text_alignment_metrics
from diff_inversion.eval.diversity import lpips_calculate_distances
from diff_inversion.eval.inversion_diagnostics import (
    latent_location_matrix,
    prediction_error_rows,
    select_evenly,
    summarize_prediction_error_rows,
    write_latent_location_plot,
    write_prediction_error_plot,
    write_rows_csv,
)
from diff_inversion.eval.normality import kl_div, kl_div2
from diff_inversion.eval.previews import write_image_comparison, write_noise_images
from diff_inversion.eval.reporting import (
    log_to_wandb,
    summarize_numeric_rows,
    write_outputs,
)
from diff_inversion.eval.sample_metrics import (
    error_distribution_metrics,
    image_pair_metrics,
    load_mask_tensor,
    latent_error_structure_metrics,
    load_rgb_tensor,
    masked_image_error_metrics,
    noise_normality_metrics,
    pair_metrics,
    patch_topk_corr,
    per_channel_pair_metrics,
    plain_area_mask,
    tensor_stats,
    trajectory_metrics,
    write_qq_plot,
)


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(tensor)!r}")
    return tensor.detach().float().cpu()


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    return getattr(config, key, default)


def _as_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(
        f"Expected latent tensor with shape [1,C,H,W] or [C,H,W], got {tuple(tensor.shape)}"
    )


def _load_optional_sample_tensor(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return _as_sample_tensor(_load_tensor(path))


def _load_noise_paths(sample_dir: Path) -> list[Path]:
    pred_noises_dir = sample_dir / "pred_noises"
    return sorted(pred_noises_dir.glob("noise_*.pt"))


def _load_latent_steps(sample_dir: Path) -> list[torch.Tensor]:
    latents_dir = sample_dir / "latents"
    latent_paths = sorted(latents_dir.glob("x_*.pt"))
    if latent_paths:
        return [_as_sample_tensor(_load_tensor(path)) for path in latent_paths]

    trajectory_path = latents_dir / "trajectory.pt"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"No latent tensors found in {latents_dir}")

    trajectory = _load_tensor(trajectory_path)
    if trajectory.ndim == 5 and trajectory.shape[1] == 1:
        return [trajectory[idx, 0] for idx in range(trajectory.shape[0])]
    if trajectory.ndim == 4:
        return [trajectory[idx] for idx in range(trajectory.shape[0])]

    raise ValueError(
        "Expected stacked trajectory with shape [T,1,C,H,W] or [T,C,H,W], "
        f"got {tuple(trajectory.shape)}"
    )


def _load_sample(
    sample_dir: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    steps = _load_latent_steps(sample_dir)
    pred_noise_paths = _load_noise_paths(sample_dir)
    pred_noises = [_as_sample_tensor(_load_tensor(path)) for path in pred_noise_paths]
    inverted_noise = _load_optional_sample_tensor(sample_dir / "inverted_noise.pt")
    metadata = _load_sample_metadata(sample_dir)
    metadata["pred_noises_count"] = len(pred_noises)
    metadata["has_inverted_noise"] = inverted_noise is not None
    return steps, pred_noises, inverted_noise, metadata


def _load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for metadata_name in ("meta.json", "prompt.json", "timesteps.json"):
        metadata_path = sample_dir / metadata_name
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata[metadata_name.removesuffix(".json")] = json.load(f)
    metadata["has_final_image"] = (sample_dir / "final.png").exists()
    metadata["has_reconstructed_image"] = (sample_dir / "reconstructed.png").exists()
    metadata["pred_noises_count"] = len(_load_noise_paths(sample_dir))
    metadata["has_initial_noise"] = (sample_dir / "initial_noise.pt").exists()
    metadata["has_inverted_noise"] = (sample_dir / "inverted_noise.pt").exists()
    return metadata


def _prompt_text(metadata: dict[str, Any]) -> str:
    prompt_record = metadata.get("prompt")
    if isinstance(prompt_record, dict):
        return str(prompt_record.get("prompt") or "")
    return ""


def _target_prompt_text(metadata: dict[str, Any]) -> str:
    prompt_record = metadata.get("prompt")
    if not isinstance(prompt_record, dict):
        return ""
    for key in ("target_prompt", "edit_prompt", "prompt_target"):
        value = prompt_record.get(key)
        if value:
            return str(value)
    return ""


def _add_inversion_metrics(
    sample_name: str,
    sample_dir: Path,
    steps: list[torch.Tensor],
    inverted_noise: torch.Tensor,
    metadata: dict[str, Any],
    max_elements: int,
    normality_sample_size: int,
    qq_num_quantiles: int,
    plain_threshold: float,
    save_noise_previews: bool,
    save_normality_plots: bool,
    noise_comparison_dir: Path,
    normality_dir: Path,
    sample_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    initial_noise = steps[0]
    inversion_error = pair_metrics(initial_noise, inverted_noise)
    inversion_error_stats = error_distribution_metrics(
        initial_noise,
        inverted_noise,
        max_elements=max_elements,
    )
    inversion_error_per_channel = per_channel_pair_metrics(initial_noise, inverted_noise)
    inversion_error_structure = latent_error_structure_metrics(
        initial_noise,
        inverted_noise,
        steps[-1],
    )
    normality_metrics = noise_normality_metrics(
        initial_noise,
        inverted_noise,
        max_elements=normality_sample_size,
        num_quantiles=qq_num_quantiles,
    )

    sample_results["inverted_noise_stats"] = tensor_stats(
        inverted_noise,
        max_elements=max_elements,
    )
    sample_results["inversion_error"] = inversion_error
    sample_results["inversion_error_stats"] = inversion_error_stats
    sample_results["inversion_error_per_channel"] = inversion_error_per_channel
    sample_results["inversion_error_structure"] = inversion_error_structure
    sample_results["noise_normality"] = normality_metrics
    region_metrics = _masked_noise_latent_region_metrics(
        sample_dir=sample_dir,
        initial_noise=initial_noise,
        inverted_noise=inverted_noise,
        plain_threshold=plain_threshold,
    )
    if region_metrics is not None:
        sample_results["plain_region_noise_latent"] = region_metrics

    noise_comparison = {
        "sample": sample_name,
        "prompt": _prompt_text(metadata),
        "input_noise_path": (sample_dir / "latents" / "x_000.pt").as_posix(),
        "inverted_noise_path": (sample_dir / "inverted_noise.pt").as_posix(),
        **inversion_error,
        **{f"error_stats/{key}": value for key, value in inversion_error_stats.items()},
        **{f"error_structure/{key}": value for key, value in inversion_error_structure.items()},
    }
    if region_metrics is not None:
        noise_comparison.update(
            {f"plain_region/{key}": value for key, value in region_metrics.items()}
        )
    if save_noise_previews:
        noise_comparison.update(
            write_noise_images(
                noise_comparison_dir / f"{sample_name}.png",
                initial_noise,
                inverted_noise,
                final_image_path=sample_dir / "final.png",
            )
        )

    normality_comparison = {
        "sample": sample_name,
        "prompt": _prompt_text(metadata),
        "input_noise_path": (sample_dir / "latents" / "x_000.pt").as_posix(),
        "inverted_noise_path": (sample_dir / "inverted_noise.pt").as_posix(),
        **normality_metrics,
    }
    if save_normality_plots:
        normality_comparison.update(
            write_qq_plot(
                normality_dir / f"{sample_name}_qq.png",
                initial_noise,
                inverted_noise,
                max_elements=normality_sample_size,
                num_quantiles=qq_num_quantiles,
                title=f"{sample_name}: initial vs inverted noise",
            )
        )
    return noise_comparison, normality_comparison, region_metrics


def _masked_mean_abs(tensor: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(tensor[:, mask].abs().mean().item())


def _masked_std(tensor: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(tensor[:, mask].std(unbiased=False).item())


def _masked_noise_latent_region_metrics(
    sample_dir: Path,
    initial_noise: torch.Tensor,
    inverted_noise: torch.Tensor,
    plain_threshold: float,
) -> dict[str, float | None] | None:
    final_image_path = sample_dir / "final.png"
    if not final_image_path.exists():
        return None

    image = load_rgb_tensor(
        final_image_path,
        size=(int(initial_noise.shape[-1]), int(initial_noise.shape[-2])),
    )
    plain_mask = plain_area_mask(image, threshold=plain_threshold)
    non_plain_mask = ~plain_mask
    delta = inverted_noise - initial_noise
    return {
        "plain_abs_error": _masked_mean_abs(delta, plain_mask),
        "non_plain_abs_error": _masked_mean_abs(delta, non_plain_mask),
        "plain_initial_noise_std": _masked_std(initial_noise, plain_mask),
        "non_plain_initial_noise_std": _masked_std(initial_noise, non_plain_mask),
        "plain_inverted_noise_std": _masked_std(inverted_noise, plain_mask),
        "non_plain_inverted_noise_std": _masked_std(inverted_noise, non_plain_mask),
        "plain_pixel_fraction": float(plain_mask.float().mean().item()),
    }


def _add_image_metrics(
    sample_name: str,
    sample_dir: Path,
    metadata: dict[str, Any],
    reconstruction_image_name: str,
    plain_threshold: float,
    save_previews: bool,
    image_comparison_dir: Path,
    sample_results: dict[str, Any],
) -> dict[str, Any] | None:
    final_image_path = sample_dir / "final.png"
    reconstructed_image_path = sample_dir / reconstruction_image_name
    if not final_image_path.exists() or not reconstructed_image_path.exists():
        return None

    metrics = image_pair_metrics(
        final_image_path,
        reconstructed_image_path,
        plain_threshold=plain_threshold,
    )
    metrics["lpips"] = None
    sample_results["reconstruction_image"] = metrics
    comparison = {
        "sample": sample_name,
        "prompt": _prompt_text(metadata),
        "source_prompt": _prompt_text(metadata),
        "sample_dir_path": sample_dir.as_posix(),
        "final_image_path": final_image_path.as_posix(),
        "reconstructed_image_path": reconstructed_image_path.as_posix(),
        "candidate_image_path": reconstructed_image_path.as_posix(),
        "reconstruction_image_name": reconstruction_image_name,
        **metrics,
    }
    target_prompt = _target_prompt_text(metadata)
    if target_prompt:
        comparison["target_prompt"] = target_prompt
    if save_previews:
        comparison.update(
            write_image_comparison(
                image_comparison_dir / f"{sample_name}.png",
                final_image_path=final_image_path,
                reconstructed_image_path=reconstructed_image_path,
                plain_threshold=plain_threshold,
            )
        )
    return comparison


def _add_edit_image_metrics(
    sample_name: str,
    sample_dir: Path,
    metadata: dict[str, Any],
    edited_image_name: str,
    plain_threshold: float,
    sample_results: dict[str, Any],
) -> dict[str, Any] | None:
    final_image_path = sample_dir / "final.png"
    edited_image_path = sample_dir / edited_image_name
    if not final_image_path.exists() or not edited_image_path.exists():
        return None

    metrics = image_pair_metrics(
        final_image_path,
        edited_image_path,
        plain_threshold=plain_threshold,
    )

    mask_path = sample_dir / "edit_mask.png"
    if mask_path.exists():
        with Image.open(final_image_path) as image:
            reference_size = image.size
        source = load_rgb_tensor(final_image_path)
        edited = load_rgb_tensor(edited_image_path, size=reference_size)
        edit_mask = load_mask_tensor(mask_path, size=reference_size)
        non_edit_mask = ~edit_mask
        delta = edited - source
        metrics.update(masked_image_error_metrics(delta, edit_mask, prefix="edit_mask"))
        metrics.update(
            masked_image_error_metrics(delta, non_edit_mask, prefix="non_edit_mask")
        )
        metrics["has_edit_mask"] = True
        metrics["edit_mask_path"] = mask_path.as_posix()
    else:
        metrics["has_edit_mask"] = False

    sample_results["edited_image"] = metrics
    comparison = {
        "sample": sample_name,
        "source_prompt": _prompt_text(metadata),
        "target_prompt": _target_prompt_text(metadata),
        "sample_dir_path": sample_dir.as_posix(),
        "final_image_path": final_image_path.as_posix(),
        "edited_image_path": edited_image_path.as_posix(),
        "edited_image_name": edited_image_name,
        **metrics,
    }
    return comparison


def _lpips_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _add_lpips_metrics(
    image_comparisons: list[dict[str, Any]],
    per_sample: dict[str, Any],
    device_name: str,
    batch_size: int,
) -> None:
    if not image_comparisons:
        return

    try:
        device = _lpips_device(device_name)
        logger.info("Calculating LPIPS on {} with batch_size={}", device, batch_size)
        from lpips import LPIPS

        loss_fn_alex = LPIPS(net="alex", version="0.1").to(device)
        for start in range(0, len(image_comparisons), batch_size):
            batch_rows = image_comparisons[start : start + batch_size]
            references = []
            candidates = []
            for row in batch_rows:
                final_image_path = Path(str(row["final_image_path"]))
                reconstructed_image_path = Path(str(row["reconstructed_image_path"]))
                with Image.open(final_image_path) as image:
                    reference_size = image.size
                references.append(load_rgb_tensor(final_image_path))
                candidates.append(load_rgb_tensor(reconstructed_image_path, size=reference_size))

            distances = lpips_calculate_distances(
                torch.stack(references),
                torch.stack(candidates),
                device=device,
                batch_size=batch_size,
                loss_fn_alex=loss_fn_alex,
            )
            for row, distance in zip(batch_rows, distances, strict=True):
                lpips_value = float(distance.item())
                row["lpips"] = lpips_value
                per_sample[row["sample"]]["reconstruction_image"]["lpips"] = lpips_value

            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("LPIPS disabled: {}", exc)


def _matrix_rows(matrix: torch.Tensor) -> list[dict[str, float | int]]:
    rows = []
    for interpolation_idx in range(matrix.shape[0]):
        row: dict[str, float | int] = {"interpolation_idx": interpolation_idx}
        for denoising_idx in range(matrix.shape[1]):
            row[f"denoising_{denoising_idx:03d}"] = float(
                matrix[interpolation_idx, denoising_idx].item()
            )
        rows.append(row)
    return rows


def run_evaluation(
    input_dir: Path,
    output_dir: Path,
    max_samples: int | None,
    patch_size: int,
    top_k: int,
    max_elements: int,
    save_noise_previews: bool,
    max_preview_samples: int | None,
    plain_threshold: float,
    normality_sample_size: int,
    qq_num_quantiles: int,
    save_normality_plots: bool,
    calculate_lpips: bool,
    lpips_device: str,
    lpips_batch_size: int,
    image_only: bool = False,
    reconstruction_image_name: str = "reconstructed.png",
    edited_image_name: str = "edited.png",
    clip_text_alignment: Any | None = None,
    inversion_diagnostics: Any | None = None,
) -> dict[str, Any]:
    sample_dirs = sorted(path for path in input_dir.glob("sample_*") if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample directories found in {input_dir}. "
            "Generate data first with `make generate_trajectories`."
        )
    total_sample_count = len(sample_dirs)
    sample_dirs = select_evenly(sample_dirs, max_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Evaluating {} of {} samples from {}",
        len(sample_dirs),
        total_sample_count,
        input_dir,
    )

    per_sample = {}
    initial_latents = []
    final_latents = []
    first_pred_noises = []
    last_pred_noises = []
    inverted_noises = []
    noise_comparisons = []
    normality_comparisons = []
    image_comparisons = []
    edit_image_comparisons = []
    clip_text_alignments = []
    masked_noise_region_comparisons = []
    noise_comparison_dir = output_dir / "noise_comparisons"
    normality_dir = output_dir / "normality"
    image_comparison_dir = output_dir / "image_comparisons"
    preview_sample_dirs = set(select_evenly(sample_dirs, max_preview_samples))
    diagnostics_enabled = not image_only and bool(
        _config_get(inversion_diagnostics, "enabled", False)
    )
    diagnostic_sample_dirs = set(
        select_evenly(sample_dirs, _config_get(inversion_diagnostics, "max_samples", 128))
        if diagnostics_enabled
        else []
    )
    diagnostics_max_elements = int(_config_get(inversion_diagnostics, "max_elements", 8192))
    diagnostics_num_interpolation_points = _config_get(
        inversion_diagnostics,
        "latent_location_num_interpolation_points",
        None,
    )
    diagnostics_target_eps_file_name = str(
        _config_get(inversion_diagnostics, "target_eps_file_name", "target_eps.pt")
    )
    diagnostics_plain_threshold = _config_get(inversion_diagnostics, "plain_threshold", None)
    if diagnostics_plain_threshold is None:
        diagnostics_plain_threshold = plain_threshold
    diagnostics_save_plots = bool(_config_get(inversion_diagnostics, "save_plots", True))
    diagnostics_dir = output_dir / "inversion_diagnostics"
    diagnostic_latent_location_matrices = []
    diagnostic_prediction_rows = []
    diagnostic_warnings = []

    for sample_dir in sample_dirs:
        save_sample_previews = save_noise_previews and sample_dir in preview_sample_dirs
        save_sample_normality_plots = save_normality_plots and sample_dir in preview_sample_dirs
        if image_only:
            steps = None
            pred_noises = []
            inverted_noise = None
            metadata = _load_sample_metadata(sample_dir)
            sample_results = {"metadata": metadata}
        else:
            steps, pred_noises, inverted_noise, metadata = _load_sample(sample_dir)
            sample_results = {
                "metadata": metadata,
                "trajectory": trajectory_metrics(steps),
                "initial_latent_stats": tensor_stats(steps[0], max_elements=max_elements),
                "final_latent_stats": tensor_stats(steps[-1], max_elements=max_elements),
            }
        per_sample[sample_dir.name] = sample_results

        if pred_noises:
            sample_results["first_pred_noise_stats"] = tensor_stats(
                pred_noises[0],
                max_elements=max_elements,
            )
            sample_results["last_pred_noise_stats"] = tensor_stats(
                pred_noises[-1],
                max_elements=max_elements,
            )
            first_pred_noises.append(pred_noises[0])
            last_pred_noises.append(pred_noises[-1])

        if inverted_noise is not None:
            noise_comparison, normality_comparison, region_metrics = _add_inversion_metrics(
                sample_name=sample_dir.name,
                sample_dir=sample_dir,
                steps=steps,
                inverted_noise=inverted_noise,
                metadata=metadata,
                max_elements=max_elements,
                normality_sample_size=normality_sample_size,
                qq_num_quantiles=qq_num_quantiles,
                plain_threshold=plain_threshold,
                save_noise_previews=save_sample_previews,
                save_normality_plots=save_sample_normality_plots,
                noise_comparison_dir=noise_comparison_dir,
                normality_dir=normality_dir,
                sample_results=sample_results,
            )
            inverted_noises.append(inverted_noise)
            noise_comparisons.append(noise_comparison)
            normality_comparisons.append(normality_comparison)
            if region_metrics is not None:
                masked_noise_region_comparisons.append(
                    {"sample": sample_dir.name, **region_metrics}
                )

            if diagnostics_enabled and sample_dir in diagnostic_sample_dirs:
                try:
                    diagnostic_latent_location_matrices.append(
                        latent_location_matrix(
                            steps=steps,
                            inverted_noise=inverted_noise,
                            max_elements=diagnostics_max_elements,
                            num_interpolation_points=diagnostics_num_interpolation_points,
                        )
                    )
                    diagnostic_prediction_rows.extend(
                        prediction_error_rows(
                            sample_dir=sample_dir,
                            steps=steps,
                            metadata=metadata,
                            target_eps_file_name=diagnostics_target_eps_file_name,
                            plain_threshold=float(diagnostics_plain_threshold),
                        )
                    )
                except Exception as exc:
                    warning = f"{sample_dir.name}: {exc}"
                    diagnostic_warnings.append(warning)
                    logger.warning(
                        "Inversion diagnostics skipped for {}: {}",
                        sample_dir.name,
                        exc,
                    )

        image_comparison = _add_image_metrics(
            sample_name=sample_dir.name,
            sample_dir=sample_dir,
            metadata=metadata,
            reconstruction_image_name=reconstruction_image_name,
            plain_threshold=plain_threshold,
            save_previews=save_sample_previews,
            image_comparison_dir=image_comparison_dir,
            sample_results=sample_results,
        )
        if image_comparison is not None:
            image_comparisons.append(image_comparison)

        edit_image_comparison = _add_edit_image_metrics(
            sample_name=sample_dir.name,
            sample_dir=sample_dir,
            metadata=metadata,
            edited_image_name=edited_image_name,
            plain_threshold=plain_threshold,
            sample_results=sample_results,
        )
        if edit_image_comparison is not None:
            edit_image_comparisons.append(edit_image_comparison)

        if steps is not None:
            initial_latents.append(steps[0])
            final_latents.append(steps[-1])

    if calculate_lpips:
        _add_lpips_metrics(
            image_comparisons,
            per_sample,
            lpips_device,
            batch_size=lpips_batch_size,
        )
    clip_text_alignments = add_clip_text_alignment_metrics(
        image_comparisons,
        per_sample,
        clip_text_alignment,
    )

    aggregate = {}
    if initial_latents:
        initial_batch = torch.stack(initial_latents)
        final_batch = torch.stack(final_latents)
        aggregate.update(
            {
                "initial_latent_stats": tensor_stats(initial_batch, max_elements=max_elements),
                "final_latent_stats": tensor_stats(final_batch, max_elements=max_elements),
                "initial_patch_topk_correlation": patch_topk_corr(
                    initial_batch, patch_size, top_k
                ),
                "final_patch_topk_correlation": patch_topk_corr(
                    final_batch, patch_size, top_k
                ),
            }
        )
    if first_pred_noises and len(first_pred_noises) == len(sample_dirs):
        aggregate["first_pred_noise_stats"] = tensor_stats(
            torch.stack(first_pred_noises),
            max_elements=max_elements,
        )
        aggregate["last_pred_noise_stats"] = tensor_stats(
            torch.stack(last_pred_noises),
            max_elements=max_elements,
        )
    if inverted_noises and len(inverted_noises) == len(sample_dirs):
        inverted_batch = torch.stack(inverted_noises)
        inversion_error_batch = inverted_batch - initial_batch
        aggregate["inverted_noise_stats"] = tensor_stats(
            inverted_batch,
            max_elements=max_elements,
        )
        aggregate["initial_vs_inverted_noise"] = pair_metrics(initial_batch, inverted_batch)
        latent_normality_corr = patch_topk_corr(
            inverted_batch,
            patch_size,
            top_k,
        )
        latent_normality_kl = kl_div(initial_batch, inverted_batch)
        latent_normality_reverse_kl = kl_div(inverted_batch, initial_batch)
        per_location_kl = kl_div2(initial_batch, inverted_batch)
        per_location_reverse_kl = kl_div2(inverted_batch, initial_batch)
        per_location_kl_x100 = per_location_kl * 100.0
        per_location_reverse_kl_x100 = per_location_reverse_kl * 100.0
        aggregate["inverted_noise_patch_topk_correlation"] = latent_normality_corr
        aggregate["latent_normality"] = {
            "corr": float(latent_normality_corr["mean"]),
            "corr_std": float(latent_normality_corr["std"]),
            "kl": float(per_location_kl_x100),
            "reverse_kl": float(per_location_reverse_kl_x100),
            "symmetric_kl": float(
                (per_location_kl_x100 + per_location_reverse_kl_x100) / 2
            ),
            "global_kl": float(latent_normality_kl),
            "global_reverse_kl": float(latent_normality_reverse_kl),
            "global_symmetric_kl": float(
                (latent_normality_kl + latent_normality_reverse_kl) / 2
            ),
            "per_location_kl": float(per_location_kl),
            "per_location_reverse_kl": float(per_location_reverse_kl),
            "per_location_symmetric_kl": float(
                (per_location_kl + per_location_reverse_kl) / 2
            ),
            "per_location_kl_x100": float(per_location_kl_x100),
            "per_location_reverse_kl_x100": float(per_location_reverse_kl_x100),
            "per_location_symmetric_kl_x100": float(
                (per_location_kl_x100 + per_location_reverse_kl_x100) / 2
            ),
        }
        aggregate["initial_vs_inverted_noise_per_channel"] = per_channel_pair_metrics(
            initial_batch,
            inverted_batch,
        )
        aggregate["inversion_error_stats"] = error_distribution_metrics(
            initial_batch,
            inverted_batch,
            max_elements=max_elements,
        )
        aggregate["inversion_error_patch_topk_correlation"] = patch_topk_corr(
            inversion_error_batch.abs(),
            patch_size,
            top_k,
        )
        aggregate["inversion_error_structure"] = latent_error_structure_metrics(
            initial_batch,
            inverted_batch,
            final_batch,
        )
    if image_comparisons:
        aggregate["final_vs_reconstructed_image"] = summarize_numeric_rows(
            image_comparisons,
            prefixes_to_skip=("reference_", "candidate_"),
        )
    if edit_image_comparisons:
        aggregate["final_vs_edited_image"] = summarize_numeric_rows(
            edit_image_comparisons,
            prefixes_to_skip=("reference_", "candidate_", "edit_mask_path"),
        )
    if clip_text_alignments:
        aggregate["clip_text_alignment"] = summarize_numeric_rows(
            clip_text_alignments,
            prefixes_to_skip=("source_", "candidate_", "target_"),
        )
    if normality_comparisons:
        aggregate["initial_vs_inverted_noise_normality"] = summarize_numeric_rows(
            normality_comparisons,
        )
    if masked_noise_region_comparisons:
        aggregate["initial_vs_inverted_noise_by_plain_region"] = summarize_numeric_rows(
            masked_noise_region_comparisons,
        )

    diagnostic_results: dict[str, Any] = {
        "enabled": diagnostics_enabled,
        "max_samples": _config_get(inversion_diagnostics, "max_samples", 128),
        "selected_samples": len(diagnostic_sample_dirs) if diagnostics_enabled else 0,
        "evaluated_samples": len(diagnostic_latent_location_matrices),
    }
    if diagnostic_warnings:
        diagnostic_results["warnings"] = diagnostic_warnings
    if diagnostic_latent_location_matrices:
        mean_latent_location = torch.stack(diagnostic_latent_location_matrices).mean(dim=0)
        latent_location_csv = diagnostics_dir / "latent_location_heatmap.csv"
        write_rows_csv(latent_location_csv, _matrix_rows(mean_latent_location))
        diagnostic_results["latent_location"] = {
            "num_samples": len(diagnostic_latent_location_matrices),
            "matrix_shape": list(mean_latent_location.shape),
            "csv_path": latent_location_csv.as_posix(),
        }
        if diagnostics_save_plots:
            heatmap_path = diagnostics_dir / "latent_location_heatmap.png"
            write_latent_location_plot(heatmap_path, mean_latent_location)
            diagnostic_results["latent_location"]["plot_path"] = heatmap_path.as_posix()
    if diagnostic_prediction_rows:
        prediction_rows_csv = diagnostics_dir / "prediction_error_samples.csv"
        write_rows_csv(prediction_rows_csv, diagnostic_prediction_rows)
        prediction_summary_rows = summarize_prediction_error_rows(diagnostic_prediction_rows)
        prediction_summary_csv = diagnostics_dir / "prediction_error_by_step.csv"
        write_rows_csv(prediction_summary_csv, prediction_summary_rows)
        diagnostic_results["prediction_error"] = {
            "num_rows": len(diagnostic_prediction_rows),
            "num_summary_rows": len(prediction_summary_rows),
            "samples_csv_path": prediction_rows_csv.as_posix(),
            "summary_csv_path": prediction_summary_csv.as_posix(),
            "by_step": prediction_summary_rows,
        }
        if diagnostics_save_plots:
            prediction_plot_path = diagnostics_dir / "prediction_error_by_step.png"
            write_prediction_error_plot(prediction_plot_path, prediction_summary_rows)
            diagnostic_results["prediction_error"]["plot_path"] = prediction_plot_path.as_posix()

    notes = [
        (
            "Image-only mode evaluates saved final and reconstructed images without "
            "loading latent trajectories or noise tensors."
            if image_only
            else "This runner evaluates saved generation trajectories."
        )
    ]
    if not image_only:
        notes.extend(
            [
                "If present, pred_noises are included as forward DDIM reference targets.",
                "If present, inverted_noise is compared against initial latent noise x_T.",
                "If present, normality diagnostics compare initial and inverted noise.",
                "Latent normality reports patch top-k correlation computed on "
                "inverted_noise; kl uses per-location Gaussian KL scaled by 100.",
                "Inversion diagnostics include latent-location and prediction-error metrics.",
            ]
        )
    notes.extend(
        [
            f"If present, {reconstruction_image_name} is compared against final.png.",
            f"If present, {edited_image_name} is compared against final.png for editing metrics.",
            "LPIPS uses the AlexNet v0.1 perceptual metric when available.",
            "Plain-area reconstruction metrics use final.png local pixel differences.",
            "CLIP target and directional metrics require target_prompt.",
            "Editing metrics need paired edited images.",
        ]
    )

    logger.info("Evaluation metrics collected; returning results")
    return {
        "input_dir": input_dir.as_posix(),
        "num_samples": len(sample_dirs),
        "total_num_samples": total_sample_count,
        "image_only": image_only,
        "num_preview_samples": len(preview_sample_dirs) if save_noise_previews else 0,
        "notes": notes,
        "aggregate": aggregate,
        "inversion_diagnostics": diagnostic_results,
        "samples": per_sample,
        "noise_comparisons": noise_comparisons,
        "normality_comparisons": normality_comparisons,
        "image_comparisons": image_comparisons,
        "edit_image_comparisons": edit_image_comparisons,
        "clip_text_alignments": clip_text_alignments,
        "masked_noise_region_comparisons": masked_noise_region_comparisons,
    }


@hydra.main(config_path="../../config", config_name="eval/run", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Evaluation config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = _resolve_path(cfg.input_dir)
    output_dir = _resolve_path(cfg.output_dir)

    results = run_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        max_samples=cfg.max_samples,
        patch_size=cfg.patch_size,
        top_k=cfg.top_k,
        max_elements=cfg.max_elements,
        save_noise_previews=cfg.save_noise_previews,
        max_preview_samples=cfg.max_preview_samples,
        plain_threshold=cfg.plain_threshold,
        normality_sample_size=cfg.normality_sample_size,
        qq_num_quantiles=cfg.qq_num_quantiles,
        save_normality_plots=cfg.save_normality_plots,
        calculate_lpips=cfg.calculate_lpips,
        lpips_device=cfg.lpips_device,
        lpips_batch_size=cfg.lpips_batch_size,
        image_only=bool(cfg.image_only),
        reconstruction_image_name=str(cfg.reconstruction_image_name),
        edited_image_name=str(cfg.edited_image_name),
        clip_text_alignment=cfg.clip_text_alignment,
        inversion_diagnostics=cfg.inversion_diagnostics,
    )
    logger.info("Evaluation returned results; writing outputs")
    write_outputs(results, output_dir)
    logger.info("Evaluation outputs written; logging to W&B")
    log_to_wandb(results, output_dir, cfg)
    logger.info("Evaluation run finished")


if __name__ == "__main__":
    main()
