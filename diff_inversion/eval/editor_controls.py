"""Small, testable controls shared by the legacy PIE-Bench editor runners."""

from __future__ import annotations

from collections.abc import Iterable

import torch


PIX2PIX_ATTENTION_LOSS_REDUCTIONS = ("legacy_sum", "per_layer_mean")
PNP_REFERENCE_PROMPT_MODES = ("empty", "source")


def pix2pix_attention_alignment_loss(
    attention_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    reduction: str,
) -> torch.Tensor:
    """Compare current/reference attention maps with an explicit layer reduction.

    legacy_sum exactly retains the inherited Pix2Pix-Zero objective: sum
    every query/token error inside a layer and then sum all layers.

    per_layer_mean first averages over every element in each attention map
    (including batch/head, query, and token axes), then gives every layer equal
    weight by averaging the layer losses. It is invariant to attention-map
    resolution and the number of instrumented layers.
    """

    if reduction not in PIX2PIX_ATTENTION_LOSS_REDUCTIONS:
        raise ValueError(
            f"Unsupported Pix2Pix-Zero attention loss reduction {reduction!r}; "
            f"expected one of {PIX2PIX_ATTENTION_LOSS_REDUCTIONS}"
        )

    layer_losses: list[torch.Tensor] = []
    for current, reference in attention_pairs:
        squared_error = (current.float() - reference.to(current.device).float()).square()
        if reduction == "legacy_sum":
            # Algebraically identical to the inherited
            # ((current - reference) ** 2).sum((1, 2)).mean(0).
            layer_loss = squared_error.sum(dim=(1, 2)).mean(dim=0)
        else:
            layer_loss = squared_error.mean()
        layer_losses.append(layer_loss)

    if not layer_losses:
        raise RuntimeError("Pix2Pix-Zero did not expose any cross-attention maps")

    stacked = torch.stack(layer_losses)
    return stacked.sum() if reduction == "legacy_sum" else stacked.mean()


def pnp_reference_embedding_index(reference_prompt_mode: str) -> int:
    """Return the branch index from get_text_embeds(source, empty).chunk(2)."""

    if reference_prompt_mode not in PNP_REFERENCE_PROMPT_MODES:
        raise ValueError(
            f"Unsupported PnP reference prompt mode {reference_prompt_mode!r}; "
            f"expected one of {PNP_REFERENCE_PROMPT_MODES}"
        )
    return 0 if reference_prompt_mode == "empty" else 1
