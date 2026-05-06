from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from diff_inversion.eval.alignment import calculate_psnr, calculate_ssim
from diff_inversion.eval.angles_distances import reconstruction_error
from diff_inversion.eval.correlation import get_top_k_corr_in_patches
from diff_inversion.eval.normality import kl_div, normal_dist_test, plt_qq, stats_from_tensor


def load_rgb_tensor(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def plain_area_mask(image: torch.Tensor, threshold: float) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image tensor, got {tuple(image.shape)}")

    diff_h = torch.abs(image[:, 1:, :] - image[:, :-1, :])
    diff_w = torch.abs(image[:, :, 1:] - image[:, :, :-1])
    diff_h = torch.nn.functional.pad(diff_h, (0, 0, 0, 1))
    diff_w = torch.nn.functional.pad(diff_w, (0, 1, 0, 0))
    diff_combined = (diff_h + diff_w) / 2
    return (diff_combined < threshold).all(dim=0)


def masked_image_error_metrics(
    delta: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, float | int | None]:
    spatial_count = int(mask.sum().item())
    total_spatial_count = int(mask.numel())
    metrics: dict[str, float | int | None] = {
        f"{prefix}_pixel_count": spatial_count,
        f"{prefix}_pixel_fraction": (
            float(spatial_count / total_spatial_count) if total_spatial_count else 0.0
        ),
    }
    if spatial_count == 0:
        metrics.update(
            {
                f"{prefix}_mse": None,
                f"{prefix}_rmse": None,
                f"{prefix}_mae": None,
                f"{prefix}_max_abs_error": None,
                f"{prefix}_mean_signed_error": None,
            }
        )
        return metrics

    masked_delta = delta[:, mask]
    mse = masked_delta.pow(2).mean()
    metrics.update(
        {
            f"{prefix}_mse": float(mse.item()),
            f"{prefix}_rmse": float(torch.sqrt(mse).item()),
            f"{prefix}_mae": float(
                reconstruction_error(torch.zeros_like(masked_delta), masked_delta)
            ),
            f"{prefix}_max_abs_error": float(masked_delta.abs().max().item()),
            f"{prefix}_mean_signed_error": float(masked_delta.mean().item()),
        }
    )
    return metrics


def image_pair_metrics(
    reference_path: Path,
    candidate_path: Path,
    plain_threshold: float,
) -> dict[str, float | int | bool | None]:
    with Image.open(reference_path) as image:
        reference_size = image.size
    with Image.open(candidate_path) as image:
        candidate_size = image.size

    reference = load_rgb_tensor(reference_path)
    candidate = load_rgb_tensor(candidate_path, size=reference_size)
    delta = candidate - reference
    mse = delta.pow(2).mean()
    rmse = torch.sqrt(mse)
    plain_mask = plain_area_mask(reference, threshold=plain_threshold)
    non_plain_mask = ~plain_mask
    reference_hwc = reference.permute(1, 2, 0).numpy()
    candidate_hwc = candidate.permute(1, 2, 0).numpy()

    metrics: dict[str, float | int | bool | None] = {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(reconstruction_error(reference, candidate)),
        "psnr_db": float(calculate_psnr(reference_hwc, candidate_hwc)),
        "ssim": float(calculate_ssim(reference_hwc, candidate_hwc)),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_signed_error": float(delta.mean().item()),
        "reference_width": int(reference_size[0]),
        "reference_height": int(reference_size[1]),
        "candidate_width": int(candidate_size[0]),
        "candidate_height": int(candidate_size[1]),
        "candidate_resized_for_metrics": bool(candidate_size != reference_size),
        "plain_threshold": float(plain_threshold),
    }
    metrics.update(masked_image_error_metrics(delta, plain_mask, prefix="plain"))
    metrics.update(masked_image_error_metrics(delta, non_plain_mask, prefix="non_plain"))
    return metrics


def flat_sample(tensor: torch.Tensor, max_elements: int) -> torch.Tensor:
    flat = tensor.detach().float().cpu().flatten()
    if flat.numel() > max_elements:
        idx = torch.linspace(0, flat.numel() - 1, max_elements).long()
        flat = flat[idx]
    return flat


def tensor_stats(tensor: torch.Tensor, max_elements: int) -> dict[str, float]:
    flat = flat_sample(tensor, max_elements)
    stats = stats_from_tensor(flat)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(1e-12)
    centered = (flat - mean) / std
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], dtype=flat.dtype),
    )
    variance = std.pow(2)
    normal_kl = 0.5 * (variance + mean.pow(2) - 1 - torch.log(variance))
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "min": float(flat.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(flat.max().item()),
        "skew": float(centered.pow(3).mean().item()),
        "excess_kurtosis": float(centered.pow(4).mean().sub(3).item()),
        "abs_mean_error_from_normal": float(mean.abs().item()),
        "abs_std_error_from_normal": float((std - 1).abs().item()),
        "normal_kl_from_standard": float(normal_kl.item()),
    }


def optional_shapiro_p_value(values: torch.Tensor) -> float | None:
    if values.numel() < 3:
        return None
    try:
        return float(normal_dist_test(values))
    except ImportError:
        return None


def noise_normality_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    max_elements: int,
    num_quantiles: int,
) -> dict[str, float | int | bool | None]:
    reference_flat = flat_sample(reference, max_elements)
    candidate_flat = flat_sample(candidate, max_elements)
    ref_to_candidate_kl = float(kl_div(reference_flat, candidate_flat))
    candidate_to_ref_kl = float(kl_div(candidate_flat, reference_flat))
    initial_shapiro = optional_shapiro_p_value(reference_flat)
    inverted_shapiro = optional_shapiro_p_value(candidate_flat)
    return {
        "sample_size": int(min(reference_flat.numel(), candidate_flat.numel())),
        "num_quantiles": int(max(2, num_quantiles)),
        "initial_shapiro_p_value": initial_shapiro,
        "inverted_shapiro_p_value": inverted_shapiro,
        "shapiro_available": initial_shapiro is not None and inverted_shapiro is not None,
        "gaussian_kl_initial_to_inverted": ref_to_candidate_kl,
        "gaussian_kl_inverted_to_initial": candidate_to_ref_kl,
        "gaussian_kl_symmetric": (ref_to_candidate_kl + candidate_to_ref_kl) / 2,
    }


def write_qq_plot(
    output_path: Path,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    max_elements: int,
    num_quantiles: int,
    title: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt_qq(
        flat_sample(reference, max_elements),
        flat_sample(candidate, max_elements),
        ds_name=title,
        t=num_quantiles,
        diff_type="initial vs inverted noise",
        path=output_path,
    )
    plt.close("all")
    return {"qq_plot_path": output_path.as_posix()}


def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.flatten(), dim=0)
    b = torch.nn.functional.normalize(b.flatten(), dim=0)
    return torch.sum(a * b)


def pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate - reference
    mse = delta.pow(2).mean()
    rmse = torch.sqrt(mse)
    reference_std = reference.std(unbiased=False).clamp_min(1e-12)
    candidate_std = candidate.std(unbiased=False).clamp_min(1e-12)
    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(reconstruction_error(reference, candidate)),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "cosine": float(safe_cosine(reference, candidate).item()),
        "relative_rmse_to_reference_std": float((rmse / reference_std).item()),
        "candidate_to_reference_std_ratio": float((candidate_std / reference_std).item()),
        "mean_shift": float((candidate.mean() - reference.mean()).item()),
    }


def per_channel_pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.ndim != candidate.ndim or reference.ndim not in (3, 4):
        return {}
    channel_dim = 0 if reference.ndim == 3 else 1
    metrics = {}
    for channel_idx in range(reference.shape[channel_dim]):
        if reference.ndim == 3:
            reference_channel = reference[channel_idx]
            candidate_channel = candidate[channel_idx]
        else:
            reference_channel = reference[:, channel_idx]
            candidate_channel = candidate[:, channel_idx]
        channel_metrics = pair_metrics(reference_channel, candidate_channel)
        for name, value in channel_metrics.items():
            metrics[f"channel_{channel_idx}/{name}"] = value
    return metrics


def error_distribution_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    max_elements: int,
) -> dict[str, float]:
    delta = candidate - reference
    flat_abs = delta.abs().flatten()
    if flat_abs.numel() > max_elements:
        idx = torch.linspace(0, flat_abs.numel() - 1, max_elements).long()
        flat_abs = flat_abs[idx]
    quantiles = torch.quantile(
        flat_abs,
        torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=flat_abs.dtype),
    )
    return {
        "signed_mean": float(delta.mean().item()),
        "signed_std": float(delta.std(unbiased=False).item()),
        "abs_mean": float(reconstruction_error(torch.zeros_like(flat_abs), flat_abs)),
        "abs_std": float(flat_abs.std(unbiased=False).item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p90": float(quantiles[1].item()),
        "abs_p95": float(quantiles[2].item()),
        "abs_p99": float(quantiles[3].item()),
        "abs_max": float(flat_abs.max().item()),
    }


def latent_error_structure_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    final_latent: torch.Tensor,
) -> dict[str, float]:
    delta = candidate - reference
    return {
        "signed_error_vs_final_latent_cosine": float(safe_cosine(delta, final_latent).item()),
        "abs_error_vs_abs_final_latent_cosine": float(
            safe_cosine(delta.abs(), final_latent.abs()).item()
        ),
    }


def trajectory_metrics(steps: list[torch.Tensor]) -> dict[str, object]:
    adjacent_l2 = []
    adjacent_mse = []
    adjacent_cosine = []

    for current, nxt in zip(steps, steps[1:]):
        delta = nxt - current
        adjacent_l2.append(torch.linalg.vector_norm(delta).item())
        adjacent_mse.append(delta.pow(2).mean().item())
        adjacent_cosine.append(safe_cosine(current, nxt).item())

    first = steps[0]
    last = steps[-1]
    first_last_delta = last - first

    def summarize(values: list[float]) -> dict[str, float]:
        values_t = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(values_t.mean().item()),
            "std": float(values_t.std(unbiased=False).item()),
            "min": float(values_t.min().item()),
            "max": float(values_t.max().item()),
        }

    return {
        "num_steps": len(steps),
        "latent_shape": list(first.shape),
        "adjacent_l2": summarize(adjacent_l2),
        "adjacent_mse": summarize(adjacent_mse),
        "adjacent_cosine": summarize(adjacent_cosine),
        "first_last_l2": float(torch.linalg.vector_norm(first_last_delta).item()),
        "first_last_mse": float(first_last_delta.pow(2).mean().item()),
        "first_last_cosine": float(safe_cosine(first, last).item()),
    }


def patch_topk_corr(tensor: torch.Tensor, patch_size: int, top_k: int) -> dict[str, float | int]:
    if tensor.shape[0] < 2:
        return {"mean": 0.0, "std": 0.0}
    return get_top_k_corr_in_patches(tensor, patch_size=patch_size, top_k=top_k)
