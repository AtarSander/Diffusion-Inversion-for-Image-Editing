"""Per-layer, per-step differences of means and CASteer concept addition."""

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

    Features have shape [step, channel]. Store raw differences; normalize each
    layer/step direction with L2 when preparing the addition offsets.
    The swap control negates half of the paired differences, chosen by null_seed;
    kth positive is paired with kth negative in source-list order. Gaussian
    directions are shared between DDIM/MW. Both controls test concept specificity.
    """
    positive = [row for row in records if row["label"] == "positive"]
    negative = [row for row in records if row["label"] == "negative"]
    if len(positive) != len(negative) or len(positive) < 2 or len(positive) % 2:
        raise ValueError(
            "Use equal, even construction class sizes >= 2 for the balanced sign null"
        )
    count = len(positive)
    rng = torch.Generator().manual_seed(null_seed)
    pair_signs = torch.tensor([1.0] * (count // 2) + [-1.0] * (count // 2))
    pair_signs = pair_signs[torch.randperm(count, generator=rng)]
    result: DirectionFit = {"directions": {}, "null_seed": null_seed}
    for rep in REPRESENTATIONS:
        result["directions"][rep] = {method: {} for method in ("DDIM", "MW")}
        names = records[0]["DDIM"]["features"][rep]
        for name in names:
            feature_shape = records[0]["DDIM"]["features"][rep][name].shape
            gaussian = torch.randn(feature_shape, generator=rng)
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
                    "swap": ((pos - neg) * pair_signs[:, None, None]).mean(0),
                    "gaussian": gaussian.clone(),
                }
    return result


def select_layers(layer_order: Sequence[str], profile: LayerProfile | str) -> list[str]:
    """Use observed forward order; do not confuse a representation with a layer subset.

    Besides the named profiles (all/last36/last12), accepts comma-separated module
    prefixes such as "mid_block" or "up_blocks.0.attentions.1"; each prefix must
    match at least one block at a dotted boundary.
    """
    if profile == "all":
        return list(layer_order)
    count = {"last36": 36, "last12": 12}.get(profile)
    if count is None:
        selected = []
        for prefix in (part.strip() for part in profile.split(",")):
            matches = [n for n in layer_order if n == prefix or n.startswith(prefix + ".")]
            if not matches:
                raise ValueError(f"Layer prefix {prefix!r} matches no captured block")
            selected += [n for n in matches if n not in selected]
        return sorted(selected, key=list(layer_order).index)
    if len(layer_order) < count:
        raise ValueError(f"Invalid layer profile {profile!r} for {len(layer_order)} blocks")
    selected = list(layer_order)[-count:]
    if any(not name.startswith("up_blocks.") for name in selected):
        raise ValueError("last36/last12 profiles require the corresponding SDXL up blocks")
    return selected


def addition_offsets(
    bundle: DirectionFit,
    representation: Representation,
    method: Method,
    control: Control,
    layers: Sequence[str],
    calls: Sequence[int],
    strength: float,
) -> Offsets:
    """Prepare alpha*s for h_new = h + alpha*s, with s = d/max(||d||_2, 1e-8).

    Addition: CASteer v4, Appendix I.2.1, Eq. (12):
    https://arxiv.org/html/2503.09630v4
    L2 normalization and its numerical floor follow compute_steering_vectors:
    https://github.com/Atmyre/CASteer/blob/main/scripts/diffusion/estimate_steering_vectors.py
    The paper's squared-norm notation differs from the released implementation;
    we follow the code's unit L2 normalization. Strength is alpha at each selected
    layer/step, independent of how many other layers/steps are selected. Zero
    directions remain zero. Applying this CA operator to SA/block is our ablation.
    """
    if not layers or not calls or not math.isfinite(strength):
        raise ValueError("A finite strength and nonempty layer/time selections are required")
    output: Offsets = {}
    for call in calls:
        output[call] = {}
        for name in layers:
            direction = bundle["directions"][representation][method][name][control][call].float()
            if not torch.isfinite(direction).all():
                raise ValueError("Finite directions required")
            norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            output[call][name] = strength * (direction / norm)
    return output
