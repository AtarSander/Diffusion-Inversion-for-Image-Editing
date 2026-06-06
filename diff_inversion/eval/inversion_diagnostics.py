from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from diff_inversion.eval.sample_metrics import load_rgb_tensor, plain_area_mask


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(tensor)!r}")
    return tensor.detach().float().cpu()


def _squeeze_sample_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"Expected [1,C,H,W] or [C,H,W], got {tuple(tensor.shape)}")


def select_evenly(items: list[Path], max_items: int | None) -> list[Path]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return items
    indices = (
        torch.linspace(0, len(items) - 1, max_items, dtype=torch.float64)
        .round()
        .long()
        .clamp_(0, len(items) - 1)
        .tolist()
    )
    return [items[idx] for idx in sorted(set(indices))]


def latent_location_matrix(
    steps: list[torch.Tensor],
    inverted_noise: torch.Tensor,
    max_elements: int,
    num_interpolation_points: int | None,
) -> torch.Tensor:
    """Distances from denoising trajectory to the noise-to-inverted-noise interpolation path."""
    if len(steps) < 2:
        raise ValueError("Expected at least two trajectory steps")

    flat_steps = torch.stack([step.flatten() for step in steps]).float()
    flat_inverted = inverted_noise.flatten().float()
    if flat_steps.shape[1] != flat_inverted.numel():
        raise ValueError(
            "Trajectory and inverted noise shapes do not match: "
            f"{flat_steps.shape[1]} vs {flat_inverted.numel()}"
        )

    if flat_steps.shape[1] > max_elements:
        element_indices = (
            torch.linspace(
                0,
                flat_steps.shape[1] - 1,
                max_elements,
                dtype=torch.float64,
            )
            .round()
            .long()
            .clamp_(0, flat_steps.shape[1] - 1)
        )
        flat_steps = flat_steps[:, element_indices]
        flat_inverted = flat_inverted[element_indices]

    initial_noise = flat_steps[0]
    num_points = int(num_interpolation_points or len(steps))
    alphas = torch.linspace(0.0, 1.0, num_points, dtype=flat_steps.dtype)
    interpolation = initial_noise.unsqueeze(0) + alphas[:, None] * (
        flat_inverted - initial_noise
    ).unsqueeze(0)

    distances_sq = (
        interpolation.pow(2).sum(dim=1, keepdim=True)
        + flat_steps.pow(2).sum(dim=1).unsqueeze(0)
        - 2 * interpolation @ flat_steps.T
    )
    distances = distances_sq.clamp_min(0.0).div(flat_steps.shape[1]).sqrt()

    column_min = distances.min(dim=0, keepdim=True).values
    column_max = distances.max(dim=0, keepdim=True).values
    return (distances - column_min) / (column_max - column_min).clamp_min(1e-12)


def _target_timesteps(metadata: dict[str, Any], target_len: int, trajectory_len: int) -> list[int]:
    timesteps = metadata.get("timesteps")
    if not isinstance(timesteps, list):
        return list(range(target_len))
    values = [int(value) for value in timesteps]
    if len(values) == trajectory_len:
        return values[1:]
    if len(values) == target_len:
        return values
    return list(range(target_len))


def _load_inversion_pred_noises(sample_dir: Path) -> tuple[list[torch.Tensor], list[int]]:
    noise_paths = sorted((sample_dir / "inversion_pred_noises").glob("noise_inv_*.pt"))
    noises = [_squeeze_sample_dim(_load_tensor(path)) for path in noise_paths]
    timesteps_path = sample_dir / "inversion_timesteps.json"
    if not timesteps_path.exists():
        return noises, list(range(len(noises)))

    with timesteps_path.open("r", encoding="utf-8") as f:
        timesteps = [int(value) for value in json.load(f)]
    if len(timesteps) == len(noises) + 1:
        timesteps = timesteps[1:]
    if len(timesteps) != len(noises):
        timesteps = list(range(len(noises)))
    return noises, timesteps


def _load_target_eps(sample_dir: Path, target_eps_file_name: str) -> torch.Tensor | None:
    path = sample_dir / "targets" / target_eps_file_name
    if not path.exists():
        return None
    target_eps = _load_tensor(path)
    if target_eps.ndim != 4:
        raise ValueError(f"Expected target eps [T,C,H,W] in {path}, got {tuple(target_eps.shape)}")
    return target_eps


def _plain_mask_for_latents(final_image_path: Path, height: int, width: int, threshold: float):
    if not final_image_path.exists():
        return None
    image = load_rgb_tensor(final_image_path, size=(width, height))
    return plain_area_mask(image, threshold=threshold)


def _masked_mean_abs(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(tensor[:, mask].abs().mean().item())


def _masked_std(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(tensor[:, mask].std(unbiased=False).item())


def prediction_error_rows(
    sample_dir: Path,
    steps: list[torch.Tensor],
    metadata: dict[str, Any],
    target_eps_file_name: str,
    plain_threshold: float,
) -> list[dict[str, float | int | str]]:
    target_eps = _load_target_eps(sample_dir, target_eps_file_name)
    if target_eps is None:
        return []

    inversion_noises, inversion_timesteps = _load_inversion_pred_noises(sample_dir)
    if not inversion_noises:
        return []

    mask = _plain_mask_for_latents(
        sample_dir / "final.png",
        height=int(target_eps.shape[-2]),
        width=int(target_eps.shape[-1]),
        threshold=plain_threshold,
    )
    if mask is None:
        return []
    non_plain_mask = ~mask

    target_timesteps = _target_timesteps(metadata, int(target_eps.shape[0]), len(steps))
    target_by_timestep = {
        int(timestep): target_eps[idx] for idx, timestep in enumerate(target_timesteps)
    }
    target_index_by_timestep = {
        int(timestep): idx for idx, timestep in enumerate(target_timesteps)
    }

    rows = []
    for inversion_step_idx, (candidate, timestep) in enumerate(
        zip(inversion_noises, inversion_timesteps, strict=False)
    ):
        reference = target_by_timestep.get(int(timestep))
        if reference is None:
            continue

        delta = candidate - reference
        plain_ref_std = _masked_std(reference, mask)
        non_plain_ref_std = _masked_std(reference, non_plain_mask)
        rows.append(
            {
                "sample": sample_dir.name,
                "inversion_step_idx": int(inversion_step_idx),
                "target_step_idx": int(target_index_by_timestep[int(timestep)]),
                "timestep": int(timestep),
                "plain_abs_error": _masked_mean_abs(delta, mask),
                "non_plain_abs_error": _masked_mean_abs(delta, non_plain_mask),
                "plain_variance_recovered_pct": (
                    100.0 * _masked_std(candidate, mask) / max(plain_ref_std, 1e-12)
                ),
                "non_plain_variance_recovered_pct": (
                    100.0
                    * _masked_std(candidate, non_plain_mask)
                    / max(non_plain_ref_std, 1e-12)
                ),
                "plain_pixel_fraction": float(mask.float().mean().item()),
            }
        )
    return rows


def summarize_prediction_error_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    by_step: dict[int, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_step.setdefault(int(row["inversion_step_idx"]), []).append(row)

    summary_rows = []
    metric_names = (
        "plain_abs_error",
        "non_plain_abs_error",
        "plain_variance_recovered_pct",
        "non_plain_variance_recovered_pct",
        "plain_pixel_fraction",
    )
    for step_idx in sorted(by_step):
        step_rows = by_step[step_idx]
        summary: dict[str, float | int] = {
            "inversion_step_idx": step_idx,
            "timestep": int(round(sum(float(row["timestep"]) for row in step_rows) / len(step_rows))),
            "target_step_idx": int(
                round(sum(float(row["target_step_idx"]) for row in step_rows) / len(step_rows))
            ),
            "num_samples": len(step_rows),
        }
        for metric_name in metric_names:
            values = torch.tensor(
                [float(row[metric_name]) for row in step_rows],
                dtype=torch.float32,
            )
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            summary[f"{metric_name}/mean"] = float(values.mean().item())
            summary[f"{metric_name}/std"] = float(values.std(unbiased=False).item())
        summary_rows.append(summary)
    return summary_rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_latent_location_plot(output_path: Path, matrix: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix.numpy(), aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Denoising step index")
    ax.set_ylabel("Interpolation point")
    ax.set_title("Trajectory distance to noise-latent interpolation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_prediction_error_plot(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = [int(row["inversion_step_idx"]) for row in rows]
    plain_error = _series(rows, "plain_abs_error/mean")
    non_plain_error = _series(rows, "non_plain_abs_error/mean")
    plain_variance = _series(rows, "plain_variance_recovered_pct/mean")
    non_plain_variance = _series(rows, "non_plain_variance_recovered_pct/mean")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_error, ax_variance) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax_error.plot(x, plain_error, color="#E69F00", label="Plain")
    ax_error.plot(x, non_plain_error, color="#6F35D4", label="Non-plain")
    ax_error.set_ylabel("Mean abs error")
    ax_error.set_title("DDIM inversion prediction error")
    ax_error.legend()

    ax_variance.plot(x, plain_variance, color="#E69F00", label="Plain")
    ax_variance.plot(x, non_plain_variance, color="#6F35D4", label="Non-plain")
    ax_variance.set_ylabel("% variance recovered")
    ax_variance.set_xlabel("Inversion step index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, float("nan"))) for row in rows]
