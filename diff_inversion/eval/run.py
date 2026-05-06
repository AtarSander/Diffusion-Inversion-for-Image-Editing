"""Run evaluation over saved SDXL trajectory samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image
import torch

from diff_inversion.eval.diversity import lpips_calculate_distances
from diff_inversion.eval.previews import write_image_comparison, write_noise_images
from diff_inversion.eval.reporting import (
    log_to_wandb,
    summarize_numeric_rows,
    write_outputs,
)
from diff_inversion.eval.sample_metrics import (
    error_distribution_metrics,
    image_pair_metrics,
    latent_error_structure_metrics,
    load_rgb_tensor,
    noise_normality_metrics,
    pair_metrics,
    patch_topk_corr,
    per_channel_pair_metrics,
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


def _load_sample(
    sample_dir: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    latents_dir = sample_dir / "latents"
    latent_paths = sorted(latents_dir.glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No latent tensors found in {latents_dir}")

    steps = [_as_sample_tensor(_load_tensor(path)) for path in latent_paths]
    pred_noise_paths = _load_noise_paths(sample_dir)
    pred_noises = [_as_sample_tensor(_load_tensor(path)) for path in pred_noise_paths]
    inverted_noise = _load_optional_sample_tensor(sample_dir / "inverted_noise.pt")
    metadata: dict[str, Any] = {}
    for metadata_name in ("meta.json", "prompt.json", "timesteps.json"):
        metadata_path = sample_dir / metadata_name
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata[metadata_name.removesuffix(".json")] = json.load(f)
    metadata["has_final_image"] = (sample_dir / "final.png").exists()
    metadata["has_reconstructed_image"] = (sample_dir / "reconstructed.png").exists()
    metadata["pred_noises_count"] = len(pred_noises)
    metadata["has_initial_noise"] = (sample_dir / "initial_noise.pt").exists()
    metadata["has_inverted_noise"] = inverted_noise is not None
    return steps, pred_noises, inverted_noise, metadata


def _prompt_text(metadata: dict[str, Any]) -> str:
    prompt_record = metadata.get("prompt")
    if isinstance(prompt_record, dict):
        return str(prompt_record.get("prompt") or "")
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
    save_noise_previews: bool,
    save_normality_plots: bool,
    noise_comparison_dir: Path,
    normality_dir: Path,
    sample_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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

    noise_comparison = {
        "sample": sample_name,
        "prompt": _prompt_text(metadata),
        "input_noise_path": (sample_dir / "latents" / "x_000.pt").as_posix(),
        "inverted_noise_path": (sample_dir / "inverted_noise.pt").as_posix(),
        **inversion_error,
        **{f"error_stats/{key}": value for key, value in inversion_error_stats.items()},
        **{f"error_structure/{key}": value for key, value in inversion_error_structure.items()},
    }
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
    return noise_comparison, normality_comparison


def _add_image_metrics(
    sample_name: str,
    sample_dir: Path,
    metadata: dict[str, Any],
    plain_threshold: float,
    save_previews: bool,
    image_comparison_dir: Path,
    sample_results: dict[str, Any],
) -> dict[str, Any] | None:
    final_image_path = sample_dir / "final.png"
    reconstructed_image_path = sample_dir / "reconstructed.png"
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
        "final_image_path": final_image_path.as_posix(),
        "reconstructed_image_path": reconstructed_image_path.as_posix(),
        **metrics,
    }
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


def _lpips_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _add_lpips_metrics(
    image_comparisons: list[dict[str, Any]],
    per_sample: dict[str, Any],
    device_name: str,
) -> None:
    if not image_comparisons:
        return

    try:
        references = []
        candidates = []
        for row in image_comparisons:
            final_image_path = Path(str(row["final_image_path"]))
            reconstructed_image_path = Path(str(row["reconstructed_image_path"]))
            with Image.open(final_image_path) as image:
                reference_size = image.size
            references.append(load_rgb_tensor(final_image_path))
            candidates.append(load_rgb_tensor(reconstructed_image_path, size=reference_size))

        device = _lpips_device(device_name)
        logger.info("Calculating LPIPS on {}", device)
        distances = lpips_calculate_distances(
            torch.stack(references),
            torch.stack(candidates),
            device=device,
        )
        for row, distance in zip(image_comparisons, distances, strict=True):
            lpips_value = float(distance.item())
            row["lpips"] = lpips_value
            per_sample[row["sample"]]["reconstruction_image"]["lpips"] = lpips_value
    except Exception as exc:
        logger.warning("LPIPS disabled: {}", exc)


def run_evaluation(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    top_k: int,
    max_elements: int,
    save_noise_previews: bool,
    plain_threshold: float,
    normality_sample_size: int,
    qq_num_quantiles: int,
    save_normality_plots: bool,
    calculate_lpips: bool,
    lpips_device: str,
) -> dict[str, Any]:
    sample_dirs = sorted(path for path in input_dir.glob("sample_*") if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample directories found in {input_dir}. "
            "Generate data first with `make generate_trajectories`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluating {} samples from {}", len(sample_dirs), input_dir)

    per_sample = {}
    initial_latents = []
    final_latents = []
    first_pred_noises = []
    last_pred_noises = []
    inverted_noises = []
    noise_comparisons = []
    normality_comparisons = []
    image_comparisons = []
    noise_comparison_dir = output_dir / "noise_comparisons"
    normality_dir = output_dir / "normality"
    image_comparison_dir = output_dir / "image_comparisons"

    for sample_dir in sample_dirs:
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
            noise_comparison, normality_comparison = _add_inversion_metrics(
                sample_name=sample_dir.name,
                sample_dir=sample_dir,
                steps=steps,
                inverted_noise=inverted_noise,
                metadata=metadata,
                max_elements=max_elements,
                normality_sample_size=normality_sample_size,
                qq_num_quantiles=qq_num_quantiles,
                save_noise_previews=save_noise_previews,
                save_normality_plots=save_normality_plots,
                noise_comparison_dir=noise_comparison_dir,
                normality_dir=normality_dir,
                sample_results=sample_results,
            )
            inverted_noises.append(inverted_noise)
            noise_comparisons.append(noise_comparison)
            normality_comparisons.append(normality_comparison)

        image_comparison = _add_image_metrics(
            sample_name=sample_dir.name,
            sample_dir=sample_dir,
            metadata=metadata,
            plain_threshold=plain_threshold,
            save_previews=save_noise_previews,
            image_comparison_dir=image_comparison_dir,
            sample_results=sample_results,
        )
        if image_comparison is not None:
            image_comparisons.append(image_comparison)

        initial_latents.append(steps[0])
        final_latents.append(steps[-1])

    if calculate_lpips:
        _add_lpips_metrics(image_comparisons, per_sample, lpips_device)

    initial_batch = torch.stack(initial_latents)
    final_batch = torch.stack(final_latents)
    aggregate = {
        "initial_latent_stats": tensor_stats(initial_batch, max_elements=max_elements),
        "final_latent_stats": tensor_stats(final_batch, max_elements=max_elements),
        "initial_patch_topk_correlation": patch_topk_corr(initial_batch, patch_size, top_k),
        "final_patch_topk_correlation": patch_topk_corr(final_batch, patch_size, top_k),
    }
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
    if normality_comparisons:
        aggregate["initial_vs_inverted_noise_normality"] = summarize_numeric_rows(
            normality_comparisons,
        )

    return {
        "input_dir": input_dir.as_posix(),
        "num_samples": len(sample_dirs),
        "notes": [
            "This runner evaluates saved generation trajectories.",
            "If present, pred_noises are included as forward DDIM reference targets.",
            "If present, inverted_noise is compared against initial latent noise x_T.",
            "If present, normality diagnostics compare initial and inverted noise.",
            "If present, reconstructed.png is compared against final.png.",
            "LPIPS uses the AlexNet v0.1 perceptual metric when available.",
            "Plain-area reconstruction metrics use final.png local pixel differences.",
            "Editing metrics need paired edited images.",
        ],
        "aggregate": aggregate,
        "samples": per_sample,
        "noise_comparisons": noise_comparisons,
        "normality_comparisons": normality_comparisons,
        "image_comparisons": image_comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/sdxl_trajectories"),
        help="Directory containing sample_*/latents/x_*.pt generated trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eval"),
        help="Directory for evaluation_summary.json and evaluation_summary.md.",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--max-elements",
        type=int,
        default=200_000,
        help="Maximum tensor elements used for distribution summaries.",
    )
    parser.add_argument(
        "--no-noise-previews",
        action="store_true",
        help="Do not write PNG previews for initial vs inverted noise comparisons.",
    )
    parser.add_argument(
        "--plain-threshold",
        type=float,
        default=0.025,
        help="Pixel-difference threshold in [0,1] for final.png plain-area masks.",
    )
    parser.add_argument(
        "--normality-sample-size",
        type=int,
        default=5_000,
        help="Maximum tensor elements per sample used for normality diagnostics.",
    )
    parser.add_argument(
        "--qq-num-quantiles",
        type=int,
        default=201,
        help="Number of quantile points used in initial-vs-inverted QQ plots.",
    )
    parser.add_argument(
        "--no-normality-plots",
        action="store_true",
        help="Compute normality metrics without writing per-sample QQ plot PNGs.",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Do not calculate LPIPS for final.png vs reconstructed.png.",
    )
    parser.add_argument(
        "--lpips-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for LPIPS. `auto` uses CUDA when available.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=os.getenv("WANDB_MODE", "disabled"),
        help="Weights & Biases logging mode.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "diff-inversion"),
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.getenv("WANDB_ENTITY"),
        help="Optional Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.getenv("WANDB_GROUP"),
        help="Optional Weights & Biases group name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.getenv("WANDB_RUN_NAME"),
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-artifact-name",
        default=os.getenv("WANDB_ARTIFACT_NAME"),
        help="Optional Weights & Biases artifact name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else None,
        help="Optional Weights & Biases tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        top_k=args.top_k,
        max_elements=args.max_elements,
        save_noise_previews=not args.no_noise_previews,
        plain_threshold=args.plain_threshold,
        normality_sample_size=args.normality_sample_size,
        qq_num_quantiles=args.qq_num_quantiles,
        save_normality_plots=not args.no_normality_plots,
        calculate_lpips=not args.no_lpips,
        lpips_device=args.lpips_device,
    )
    write_outputs(results, args.output_dir)
    log_to_wandb(results, args.output_dir, args)


if __name__ == "__main__":
    main()
