from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_top_k_corr(tensor: torch.Tensor, top_k: int = 10) -> dict:
    assert tensor.ndim == 4, "Tensor must have 4 dimensions: [batch_size, channels, height, width]"
    corr_matrix = torch.corrcoef(tensor.flatten(1, 3).T)
    corr_matrix = corr_matrix.triu(diagonal=1)
    top_k_coeffs = torch.topk(corr_matrix.abs().flatten(), top_k)
    top_k_values = top_k_coeffs[0]

    return {"mean": top_k_values.mean().item(), "std": top_k_values.std().item()}


def get_top_k_corr_in_patches(tensor: torch.Tensor, patch_size: int = 8, top_k: int = 10) -> dict:
    assert tensor.ndim == 4, "Tensor must have 4 dimensions: [batch_size, channels, height, width]"
    num_examples, channels, height, width = tensor.shape

    avg_topk_values = []
    patch_counter = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patches = tensor[:, :, i : i + patch_size, j : j + patch_size].reshape(
                num_examples, -1
            )

            if patches.size(1) > 1:  # Ensure there are at least two elements
                mean = patches.mean(dim=0, keepdim=True)
                std = patches.std(dim=0, keepdim=True)
                normalized_patches = (patches - mean) / (std + 1e-5)

                corr_matrix = torch.corrcoef(normalized_patches.T)

                # Extract the upper triangle of the correlation matrix, excluding the diagonal
                triu_indices = torch.triu_indices(patches.size(1), patches.size(1), offset=1)
                upper_triangle_values = corr_matrix[triu_indices[0], triu_indices[1]]

                # Get the top-k absolute correlation coefficients for this patch
                top_k_values, _ = torch.topk(
                    upper_triangle_values.abs(), min(top_k, upper_triangle_values.numel())
                )
                avg_topk_values.append(top_k_values)
            else:
                patch_counter += 1

    if avg_topk_values:
        avg_topk_values = torch.cat(avg_topk_values, dim=0)
        mean_top_k = avg_topk_values.mean().item()
        std_top_k = avg_topk_values.std().item()
    else:
        mean_top_k = 0
        std_top_k = 0

    if patch_counter > 0:
        logger.warning("{} patches were empty", patch_counter)

    return {"mean": mean_top_k, "std": std_top_k}


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc


def _load_tensor(path: Path, dtype: torch.dtype, weights_only: bool) -> torch.Tensor:
    logger.debug("Loading tensor from {}", path)
    return torch.load(path, weights_only=weights_only).to(dtype)


@hydra.main(config_path="../../config/eval", config_name="correlation", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Correlation config:\n{}", OmegaConf.to_yaml(cfg))

    outputs_dir = _resolve_path(cfg.outputs_dir)
    dtype = _resolve_torch_dtype(cfg.dtype)
    metrics = cfg.metrics

    for model in cfg.models:
        model_dir = outputs_dir / model / cfg.timestep_dir
        logger.info("Running correlation for model '{}' from {}", model, model_dir)

        noisess = _load_tensor(model_dir / cfg.filenames.noise, dtype, cfg.weights_only)
        latents = _load_tensor(model_dir / cfg.filenames.latents, dtype, cfg.weights_only)
        samples = _load_tensor(model_dir / cfg.filenames.samples, dtype, cfg.weights_only)

        logger.info(
            "Tensor dtypes: noisess={}, latents={}, samples={}",
            noisess.dtype,
            latents.dtype,
            samples.dtype,
        )
        logger.info(
            "Tensor shapes: noisess={}, latents={}, samples={}",
            tuple(noisess.shape),
            tuple(latents.shape),
            tuple(samples.shape),
        )

        logger.info(
            "Noisess: {}",
            get_top_k_corr_in_patches(
                noisess,
                top_k=metrics.top_k,
                patch_size=metrics.patch_size,
            ),
        )
        logger.info(
            "Latents: {}",
            get_top_k_corr_in_patches(
                latents,
                top_k=metrics.top_k,
                patch_size=metrics.patch_size,
            ),
        )
        logger.info(
            "Samples: {}",
            get_top_k_corr_in_patches(
                samples,
                top_k=metrics.top_k,
                patch_size=metrics.patch_size,
            ),
        )


if __name__ == "__main__":
    main()
