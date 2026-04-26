"""Run lightweight evaluation over saved SDXL trajectory samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw
import torch


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


def _tensor_stats(tensor: torch.Tensor, max_elements: int) -> dict[str, float]:
    flat = tensor.flatten()
    if flat.numel() > max_elements:
        idx = torch.linspace(0, flat.numel() - 1, max_elements).long()
        flat = flat[idx]

    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(1e-12)
    centered = (flat - mean) / std
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "skew": float(centered.pow(3).mean().item()),
        "excess_kurtosis": float(centered.pow(4).mean().sub(3).item()),
        "abs_mean_error_from_normal": float(mean.abs().item()),
        "abs_std_error_from_normal": float((std - 1).abs().item()),
    }


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.flatten(), dim=0)
    b = torch.nn.functional.normalize(b.flatten(), dim=0)
    return torch.sum(a * b)


def _pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate - reference
    return {
        "mse": float(delta.pow(2).mean().item()),
        "mae": float(delta.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "cosine": float(_safe_cosine(reference, candidate).item()),
    }


def _normalize_to_uint8(tensor: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    if vmax <= vmin:
        return torch.zeros_like(tensor, dtype=torch.uint8)
    normalized = (tensor - vmin) / (vmax - vmin)
    return normalized.clamp(0, 1).mul(255).round().to(torch.uint8)


def _channel_grid_image(tensor: torch.Tensor, vmin: float, vmax: float) -> Image.Image:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W] tensor for preview, got {tuple(tensor.shape)}")

    channels = min(int(tensor.shape[0]), 4)
    height = int(tensor.shape[1])
    width = int(tensor.shape[2])
    canvas = Image.new("L", (width * 2, height * 2), color=0)

    for channel_idx in range(channels):
        channel = _normalize_to_uint8(tensor[channel_idx], vmin, vmax)
        image = Image.fromarray(channel.numpy(), mode="L")
        canvas.paste(image, ((channel_idx % 2) * width, (channel_idx // 2) * height))

    return canvas.convert("RGB")


def _write_noise_images(
    output_path: Path,
    initial_noise: torch.Tensor,
    inverted_noise: torch.Tensor,
) -> dict[str, str]:
    combined = torch.cat([initial_noise.flatten(), inverted_noise.flatten()])
    vmin = float(combined.quantile(0.01).item())
    vmax = float(combined.quantile(0.99).item())
    error = (inverted_noise - initial_noise).abs()
    error_vmax = float(error.quantile(0.99).item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = output_path.with_name(f"{output_path.stem}_input_noise.png")
    inverted_path = output_path.with_name(f"{output_path.stem}_inverted_noise.png")
    error_path = output_path.with_name(f"{output_path.stem}_abs_error.png")

    input_image = _channel_grid_image(initial_noise, vmin, vmax)
    inverted_image = _channel_grid_image(inverted_noise, vmin, vmax)
    error_image = _channel_grid_image(error, 0.0, max(error_vmax, 1e-8))

    input_image.save(input_path)
    inverted_image.save(inverted_path)
    error_image.save(error_path)

    panels = [
        ("input: latents/x_000.pt", input_image),
        ("inverted: inverted_noise.pt", inverted_image),
        ("abs error", error_image),
    ]

    label_height = 18
    panel_width, panel_height = panels[0][1].size
    canvas = Image.new(
        "RGB", (panel_width * len(panels), panel_height + label_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        x = idx * panel_width
        draw.text((x + 4, 3), label, fill="black")
        canvas.paste(image, (x, label_height))

    canvas.save(output_path)
    return {
        "preview_path": output_path.as_posix(),
        "input_noise_image_path": input_path.as_posix(),
        "inverted_noise_image_path": inverted_path.as_posix(),
        "abs_error_image_path": error_path.as_posix(),
    }


def _trajectory_metrics(steps: list[torch.Tensor]) -> dict[str, Any]:
    adjacent_l2 = []
    adjacent_mse = []
    adjacent_cosine = []

    for current, nxt in zip(steps, steps[1:]):
        delta = nxt - current
        adjacent_l2.append(torch.linalg.vector_norm(delta).item())
        adjacent_mse.append(delta.pow(2).mean().item())
        adjacent_cosine.append(_safe_cosine(current, nxt).item())

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
        "first_last_cosine": float(_safe_cosine(first, last).item()),
    }


def _patch_topk_corr(tensors: torch.Tensor, patch_size: int, top_k: int) -> dict[str, float | int]:
    if tensors.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] tensor, got {tuple(tensors.shape)}")
    if tensors.shape[0] < 2:
        return {"mean": 0.0, "std": 0.0, "num_values": 0}

    _, _, height, width = tensors.shape
    values = []
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = tensors[:, :, row : row + patch_size, col : col + patch_size].reshape(
                tensors.shape[0],
                -1,
            )
            if patch.shape[1] < 2:
                continue
            std = patch.std(dim=0, unbiased=False)
            keep = std > 1e-8
            patch = patch[:, keep]
            if patch.shape[1] < 2:
                continue

            patch = (patch - patch.mean(dim=0, keepdim=True)) / patch.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            ).clamp_min(1e-8)
            corr = patch.T @ patch / patch.shape[0]
            corr = torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            triu = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
            upper = corr[triu[0], triu[1]].abs()
            if upper.numel() == 0:
                continue
            values.append(torch.topk(upper, min(top_k, upper.numel())).values)

    if not values:
        return {"mean": 0.0, "std": 0.0, "num_values": 0}

    values_t = torch.cat(values)
    return {
        "mean": float(values_t.mean().item()),
        "std": float(values_t.std(unbiased=False).item()),
        "num_values": int(values_t.numel()),
    }


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
    metadata["pred_noises_count"] = len(pred_noises)
    metadata["has_initial_noise"] = (sample_dir / "initial_noise.pt").exists()
    metadata["has_inverted_noise"] = inverted_noise is not None
    return steps, pred_noises, inverted_noise, metadata


def run_evaluation(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    top_k: int,
    max_elements: int,
    save_noise_previews: bool,
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
    noise_comparison_dir = output_dir / "noise_comparisons"

    for sample_dir in sample_dirs:
        steps, pred_noises, inverted_noise, metadata = _load_sample(sample_dir)
        per_sample[sample_dir.name] = {
            "metadata": metadata,
            "trajectory": _trajectory_metrics(steps),
            "initial_latent_stats": _tensor_stats(steps[0], max_elements=max_elements),
            "final_latent_stats": _tensor_stats(steps[-1], max_elements=max_elements),
        }
        if pred_noises:
            per_sample[sample_dir.name]["first_pred_noise_stats"] = _tensor_stats(
                pred_noises[0],
                max_elements=max_elements,
            )
            per_sample[sample_dir.name]["last_pred_noise_stats"] = _tensor_stats(
                pred_noises[-1],
                max_elements=max_elements,
            )
            first_pred_noises.append(pred_noises[0])
            last_pred_noises.append(pred_noises[-1])
        if inverted_noise is not None:
            inversion_error = _pair_metrics(steps[0], inverted_noise)
            per_sample[sample_dir.name]["inverted_noise_stats"] = _tensor_stats(
                inverted_noise,
                max_elements=max_elements,
            )
            per_sample[sample_dir.name]["inversion_error"] = inversion_error
            inverted_noises.append(inverted_noise)
            comparison = {
                "sample": sample_dir.name,
                "input_noise_path": (sample_dir / "latents" / "x_000.pt").as_posix(),
                "inverted_noise_path": (sample_dir / "inverted_noise.pt").as_posix(),
                **inversion_error,
            }
            if save_noise_previews:
                preview_path = noise_comparison_dir / f"{sample_dir.name}.png"
                comparison.update(_write_noise_images(preview_path, steps[0], inverted_noise))
            noise_comparisons.append(comparison)
        initial_latents.append(steps[0])
        final_latents.append(steps[-1])

    initial_batch = torch.stack(initial_latents)
    final_batch = torch.stack(final_latents)
    aggregate = {
        "initial_latent_stats": _tensor_stats(initial_batch, max_elements=max_elements),
        "final_latent_stats": _tensor_stats(final_batch, max_elements=max_elements),
        "initial_patch_topk_correlation": _patch_topk_corr(initial_batch, patch_size, top_k),
        "final_patch_topk_correlation": _patch_topk_corr(final_batch, patch_size, top_k),
    }
    if first_pred_noises and len(first_pred_noises) == len(sample_dirs):
        aggregate["first_pred_noise_stats"] = _tensor_stats(
            torch.stack(first_pred_noises),
            max_elements=max_elements,
        )
        aggregate["last_pred_noise_stats"] = _tensor_stats(
            torch.stack(last_pred_noises),
            max_elements=max_elements,
        )
    if inverted_noises and len(inverted_noises) == len(sample_dirs):
        inverted_batch = torch.stack(inverted_noises)
        aggregate["inverted_noise_stats"] = _tensor_stats(
            inverted_batch,
            max_elements=max_elements,
        )
        aggregate["initial_vs_inverted_noise"] = _pair_metrics(initial_batch, inverted_batch)

    results = {
        "input_dir": input_dir.as_posix(),
        "num_samples": len(sample_dirs),
        "notes": [
            "This runner evaluates saved generation trajectories.",
            "If present, pred_noises are included as forward DDIM reference targets.",
            "If present, inverted_noise is compared against initial latent noise x_T.",
            "Reconstruction/editing metrics need paired reconstructed or edited images.",
        ],
        "aggregate": aggregate,
        "samples": per_sample,
        "noise_comparisons": noise_comparisons,
    }
    return results


def _flatten_metrics(
    data: dict[str, Any], prefix: str = ""
) -> dict[str, float | int | str | bool]:
    flat: dict[str, float | int | str | bool] = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key))
        elif isinstance(value, (float, int, str, bool)):
            flat[full_key] = value
    return flat


def _build_samples_table_rows(
    results: dict[str, Any],
) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for sample_name, sample in results["samples"].items():
        row: dict[str, float | int | str | bool] = {"sample": sample_name}
        row.update(_flatten_metrics(sample["metadata"], prefix="metadata"))
        row.update(_flatten_metrics(sample["trajectory"], prefix="trajectory"))
        row.update(_flatten_metrics(sample["initial_latent_stats"], prefix="initial_latent"))
        row.update(_flatten_metrics(sample["final_latent_stats"], prefix="final_latent"))
        if "first_pred_noise_stats" in sample:
            row.update(
                _flatten_metrics(sample["first_pred_noise_stats"], prefix="first_pred_noise")
            )
        if "last_pred_noise_stats" in sample:
            row.update(_flatten_metrics(sample["last_pred_noise_stats"], prefix="last_pred_noise"))
        if "inverted_noise_stats" in sample:
            row.update(_flatten_metrics(sample["inverted_noise_stats"], prefix="inverted_noise"))
        if "inversion_error" in sample:
            row.update(_flatten_metrics(sample["inversion_error"], prefix="inversion_error"))
        rows.append(row)
    return rows


def _sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return sanitized.strip("-.") or "evaluation-summary"


def log_to_wandb(results: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return

    try:
        import wandb
    except ImportError:
        logger.warning(
            "W&B logging skipped because `wandb` is not installed in the active environment"
        )
        return

    wandb_root = output_dir / "wandb"
    cache_dir = wandb_root / "cache"
    config_dir = wandb_root / "config"
    data_dir = wandb_root / "data"
    files_dir = wandb_root / "files"
    for directory in (wandb_root, cache_dir, config_dir, data_dir, files_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", wandb_root.as_posix())
    os.environ.setdefault("WANDB_CACHE_DIR", cache_dir.as_posix())
    os.environ.setdefault("WANDB_CONFIG_DIR", config_dir.as_posix())
    os.environ.setdefault("WANDB_DATA_DIR", data_dir.as_posix())

    try:
        config = {
            "input_dir": results["input_dir"],
            "output_dir": output_dir.as_posix(),
            "num_samples": results["num_samples"],
            "patch_size": args.patch_size,
            "top_k": args.top_k,
            "max_elements": args.max_elements,
        }
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            job_type="evaluation",
            mode=args.wandb_mode,
            config=config,
            dir=wandb_root.as_posix(),
            settings=wandb.Settings(
                root_dir=wandb_root.as_posix(),
                x_files_dir=files_dir.as_posix(),
                x_disable_stats=True,
            ),
        )

        aggregate_metrics = _flatten_metrics(results["aggregate"], prefix="aggregate")
        aggregate_metrics["num_samples"] = results["num_samples"]
        wandb.log(aggregate_metrics)

        rows = _build_samples_table_rows(results)
        if rows:
            columns = sorted({key for row in rows for key in row})
            table = wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*[row.get(column) for column in columns])
            wandb.log({"samples_table": table})

        summary_json = output_dir / "evaluation_summary.json"
        summary_md = output_dir / "evaluation_summary.md"
        artifact = wandb.Artifact(
            name=_sanitize_artifact_name(args.wandb_artifact_name or output_dir.name),
            type="evaluation",
        )
        if summary_json.exists():
            artifact.add_file(summary_json.as_posix())
        if summary_md.exists():
            artifact.add_file(summary_md.as_posix())
        noise_comparison_dir = output_dir / "noise_comparisons"
        if noise_comparison_dir.exists():
            artifact.add_dir(noise_comparison_dir.as_posix(), name="noise_comparisons")
        run.log_artifact(artifact)

        run.summary["input_dir"] = results["input_dir"]
        run.summary["notes"] = results["notes"]
        run.finish()
        logger.success("Logged evaluation to Weights & Biases")
    except Exception as exc:
        logger.warning("W&B logging skipped because initialization failed: {}", exc)


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "evaluation_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    noise_comparisons = results.get("noise_comparisons") or []
    if noise_comparisons:
        comparison_dir = output_dir / "noise_comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_json_path = comparison_dir / "noise_comparisons.json"
        comparison_csv_path = comparison_dir / "noise_comparisons.csv"
        with comparison_json_path.open("w", encoding="utf-8") as f:
            json.dump(noise_comparisons, f, indent=2)
        columns = sorted({key for row in noise_comparisons for key in row})
        with comparison_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(noise_comparisons)

    aggregate = results["aggregate"]
    markdown_path = output_dir / "evaluation_summary.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"- Input directory: `{results['input_dir']}`\n")
        f.write(f"- Samples: {results['num_samples']}\n\n")
        if noise_comparisons:
            f.write("- Noise comparison artifacts: `noise_comparisons/`\n\n")
        f.write("## Aggregate\n\n")
        for name, metrics in aggregate.items():
            f.write(f"### {name}\n\n")
            for key, value in metrics.items():
                f.write(f"- `{key}`: {value}\n")
            f.write("\n")
        f.write("## Notes\n\n")
        for note in results["notes"]:
            f.write(f"- {note}\n")

    logger.success("Saved {}", json_path)
    logger.success("Saved {}", markdown_path)


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
    parser.add_argument("--top-k", type=int, default=10)
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
    )
    write_outputs(results, args.output_dir)
    log_to_wandb(results, args.output_dir, args)


if __name__ == "__main__":
    main()
