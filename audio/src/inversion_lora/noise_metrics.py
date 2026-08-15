# ABOUTME: Distributional checks on inverted latents: KL to a Gaussian reference and top-k
# ABOUTME: cross-dimension correlation within latent patches, each against a measured null.

import numpy as np
import torch
import torch.distributions as dist


def kl_div_scalar(reference: torch.Tensor, latents: torch.Tensor) -> float:
    """KL between single Gaussians fitted to each whole tensor.

    Both tensors collapse to one (mean, std) pair, so this only sees a global scale or shift and
    is insensitive to per-dimension structure. `kl_div_per_dim` covers that.

    Args:
        reference: Gaussian reference sample, any shape.
        latents: Inverted latents, any shape.

    Returns:
        KL(reference || latents) in nats.
    """
    p = dist.Normal(reference.mean().float(), reference.std().float())
    q = dist.Normal(latents.mean().float(), latents.std().float())
    return float(dist.kl_divergence(p, q))


def kl_div_per_dim(reference: torch.Tensor, latents: torch.Tensor) -> float:
    """KL between per-dimension Gaussians estimated across the batch, averaged over dimensions.

    Args:
        reference: Gaussian reference sample `[N, C, H, W]`.
        latents: Inverted latents `[N, C, H, W]`.

    Returns:
        KL(reference || latents) summed over dimensions and divided by their count.
    """
    assert reference.ndim == 4 and latents.ndim == 4, (reference.shape, latents.shape)
    assert reference.shape[1:] == latents.shape[1:], (reference.shape, latents.shape)

    p = dist.Normal(reference.mean(dim=0).float(), reference.std(dim=0).float() + 1e-6)
    q = dist.Normal(latents.mean(dim=0).float(), latents.std(dim=0).float() + 1e-6)
    divergence = dist.kl_divergence(p, q).sum()
    return float(divergence / np.prod(latents.shape[1:]))


def top_k_corr_in_patches(
    latents: torch.Tensor, patch_size: int = 8, top_k: int = 20
) -> dict[str, float]:
    """Mean of the largest absolute cross-dimension correlations inside each latent patch.

    Correlations are taken across the batch between pairs of positions within a patch, so iid
    noise scores near zero and any spatial structure the inversion leaves behind shows up as a
    larger value. Patches keep the correlation matrix small: the full latent would need a
    32768x32768 matrix.

    The value has a positive floor that grows as the batch shrinks (top-k of many sample
    correlations each with sd ~1/sqrt(N)), so it is only meaningful against a Gaussian reference
    scored at the same batch size.

    Args:
        latents: Latents `[N, C, H, W]`, at least 2 examples.
        patch_size: Side of the square patch over the last two dimensions.
        top_k: How many of the largest absolute correlations to average per patch.

    Returns:
        Mean and std of the pooled top-k absolute correlations.
    """
    assert latents.ndim == 4, f"expected [N, C, H, W], got {tuple(latents.shape)}"
    assert latents.shape[0] > 1, "correlation across the batch needs at least 2 examples"
    num_examples, _, height, width = latents.shape
    latents = latents.detach().float()

    collected: list[torch.Tensor] = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = latents[:, :, i : i + patch_size, j : j + patch_size].reshape(num_examples, -1)
            if patch.shape[1] <= 1:
                continue
            corr = torch.corrcoef(patch.T)
            rows, cols = torch.triu_indices(patch.shape[1], patch.shape[1], offset=1)
            values = corr[rows, cols].abs()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            collected.append(torch.topk(values, min(top_k, values.numel())).values)

    assert collected, f"no usable patch at patch_size={patch_size} for {tuple(latents.shape)}"
    pooled = torch.cat(collected)
    return {"mean": float(pooled.mean()), "std": float(pooled.std())}


def noise_report(
    latents: torch.Tensor,
    reference: torch.Tensor,
    prefix: str,
    reference_is_ground_truth: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Full distributional comparison of inverted latents against a Gaussian reference.

    A Shapiro-Wilk normality rate used to live here and was removed: it assumes iid samples,
    while latent elements are spatially correlated, so it is not calibrated on this data.
    Feeding it samples that are marginally exactly N(0,1) but spatially smoothed drives the
    rejection rate from 4% to 29% purely from correlation. `corr_topk` measures that correlation
    directly, against a measured null, and the KL terms cover the distributional match.

    `corr_topk` and `kl_per_dim` both estimate per-dimension statistics across the batch, so both
    carry a positive floor that grows as the batch shrinks: at N=2 every correlation is exactly
    1.0 and `kl_per_dim` is enormous, whatever the latents look like. Each is therefore reported
    alongside a `_reference` value measured on an independent Gaussian draw at the same batch
    size, which is the null this run should be read against. Comparing the raw numbers across
    runs is only valid at equal N.

    Args:
        latents: Inverted latents `[N, C, H, W]`.
        reference: Reference sample of the same shape. For generated audio this is the actual
            initial noise the sample came from; for real audio no such noise exists, so the
            caller passes a fresh standard normal draw.
        prefix: Metric name prefix, e.g. `eval/generated`.
        reference_is_ground_truth: True when `reference` is the exact noise the sample came from,
            which makes an elementwise error meaningful; false for a random stand-in.
        seed: Seed for the independent draw used to measure the floors.

    Returns:
        Flat dict of metric name to value.
    """
    assert latents.shape == reference.shape, (latents.shape, reference.shape)
    null_draw = torch.randn(
        latents.shape, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float32
    )
    latent_corr = top_k_corr_in_patches(latents)
    out = {
        f"{prefix}/kl_scalar": kl_div_scalar(reference, latents),
        f"{prefix}/kl_per_dim": kl_div_per_dim(reference, latents),
        f"{prefix}/kl_per_dim_reference": kl_div_per_dim(reference, null_draw),
        f"{prefix}/corr_topk": latent_corr["mean"],
        f"{prefix}/corr_topk_std": latent_corr["std"],
        f"{prefix}/corr_topk_reference": top_k_corr_in_patches(null_draw)["mean"],
        f"{prefix}/latent_mean": float(latents.mean()),
        f"{prefix}/latent_std": float(latents.std()),
    }
    if reference_is_ground_truth:
        # The sharpest available signal: inversion of a generated sample should return the very
        # noise it was sampled from, not merely something Gaussian.
        out[f"{prefix}/noise_mse"] = float(
            torch.mean((latents.float() - reference.float()) ** 2)
        )
    return out
