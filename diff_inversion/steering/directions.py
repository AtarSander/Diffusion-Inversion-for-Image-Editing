"""Per-layer, per-denoising-step differences of means and common dose scaling."""

import math
from collections.abc import Sequence

import torch

from .hooks import REPRESENTATIONS
from .types import (
    Control,
    DirectionFit,
    LayerProfile,
    Method,
    Offsets,
    Representation,
    SourceRecord,
)


def fit_directions(records: Sequence[SourceRecord], *, null_seed: int = 123) -> DirectionFit:
    """Fit on construction images only. Each record holds both DDIM and MW reads.

    Features have shape [step, channel]; moments are measured before pooling.
    A balanced sign swap is a control on paired class differences. Gaussian
    orientations are shared between methods. Neither null is a style vector.
    """
    positive = [row for row in records if row["label"] == "positive"]
    negative = [row for row in records if row["label"] == "negative"]
    if len(positive) != len(negative) or len(positive) < 2 or len(positive) % 2:
        raise ValueError(
            "Use equal, even construction class sizes >= 2 for the balanced sign null"
        )
    count = len(positive)
    rng = torch.Generator().manual_seed(null_seed)
    signs = torch.tensor([1.0] * (count // 2) + [-1.0] * (count // 2))
    signs = signs[torch.randperm(count, generator=rng)]
    result: DirectionFit = {"directions": {}, "reference_rms": {}, "null_seed": null_seed}
    for rep in REPRESENTATIONS:
        result["directions"][rep] = {method: {} for method in ("DDIM", "MW")}
        result["reference_rms"][rep] = {}
        names = records[0]["DDIM"]["features"][rep]
        for name in names:
            means = torch.stack(
                [
                    row[method]["moments"][rep][name].double()
                    for row in records
                    for method in ("DDIM", "MW")
                ]
            )
            if not torch.isfinite(means).all() or bool((means < 0).any()):
                raise ValueError("Expected finite, nonnegative full-map second moments")
            result["reference_rms"][rep][name] = means.mean(0).sqrt().float()
            random = torch.randn(records[0]["DDIM"]["features"][rep][name].shape, generator=rng)
            for method in ("DDIM", "MW"):
                pos = torch.stack([row[method]["features"][rep][name].float() for row in positive])
                neg = torch.stack([row[method]["features"][rep][name].float() for row in negative])
                if (
                    pos.shape != neg.shape
                    or pos.ndim != 3
                    or not torch.isfinite(pos).all()
                    or not torch.isfinite(neg).all()
                ):
                    raise ValueError("Expected matched finite [image, step, channel] features")
                result["directions"][rep][method][name] = {
                    "true": pos.mean(0) - neg.mean(0),
                    "swap": ((pos - neg) * signs[:, None, None]).mean(0),
                    "gaussian": random.clone(),
                }
    return result


def select_layers(layer_order: Sequence[str], profile: LayerProfile) -> list[str]:
    """Use observed forward order; do not confuse a representation with a layer subset."""
    if profile == "all":
        return list(layer_order)
    count = {"last36": 36, "last12": 12}.get(profile)
    if count is None or len(layer_order) < count:
        raise ValueError(f"Invalid layer profile {profile!r} for {len(layer_order)} blocks")
    selected = list(layer_order)[-count:]
    if any(not name.startswith("up_blocks.") for name in selected):
        raise ValueError("last36/last12 profiles require the corresponding SDXL up blocks")
    return selected


def scaled_offsets(
    bundle: DirectionFit,
    representation: Representation,
    method: Method,
    control: Control,
    layers: Sequence[str],
    calls: Sequence[int],
    strength: float,
) -> Offsets:
    """Apply lambda*sqrt(10/(L*K))*reference_RMS*d/max(RMS(d), 1e-8).

    This is our dose convention, not a CASteer formula. The factor 10 preserves
    the historical reference of one CA location over ten calls. For nonzero
    directions above the norm floor, the requested squared relative RMS summed
    over L layers and K calls is 10*lambda**2, before delivery rounding.
    This does not equalize perceptual changes. See the 2026-10-13 entry in
    docs/experiments/wikiart_style_steering_20261012.md.
    """
    if not layers or not calls or not math.isfinite(strength):
        raise ValueError("A finite strength and nonempty layer/time selections are required")
    amplitude = strength * math.sqrt(10.0 / (len(layers) * len(calls)))
    output: Offsets = {}
    for call in calls:
        output[call] = {}
        for name in layers:
            direction = bundle["directions"][representation][method][name][control][call].float()
            rms = float(bundle["reference_rms"][representation][name][call])
            if not torch.isfinite(direction).all() or not math.isfinite(rms) or rms < 0:
                raise ValueError("Finite directions and nonnegative reference RMS required")
            # Preserve the original diagnostic FP64 RMS, rather than a pooled-map norm.
            norm = float(direction.double().square().mean().sqrt())
            output[call][name] = direction * (amplitude * rms / max(norm, 1e-8))
    return output
