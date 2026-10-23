"""Read CA/SA/block updates and add offsets to the conditional branch."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from diffusers.models.attention import BasicTransformerBlock
from torch.utils.hooks import RemovableHandle

from .types import FeatureMaps, LayerTensors, Representation

REPRESENTATIONS: tuple[Representation, ...] = ("CA", "SA", "block")


def transformer_blocks(unet: torch.nn.Module) -> dict[str, BasicTransformerBlock]:
    """Find individual transformer blocks, rather than whole UNet up/down blocks."""
    blocks = {
        name: module
        for name, module in unet.named_modules()
        if isinstance(module, BasicTransformerBlock)
    }
    if not blocks:
        raise ValueError("No BasicTransformerBlock found in the UNet")
    for name, block in blocks.items():
        if (
            block.only_cross_attention
            or block.double_self_attention
            or block.attn1 is None
            or block.attn2 is None
        ):
            raise ValueError(f"{name} must contain SA, CA and feed-forward layers")
    return blocks


def conditional_branch(tensor: torch.Tensor) -> torch.Tensor:
    """The runtime processes one image: batch 0 is unconditional, batch 1 conditional."""
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError("Expected [2, token, channel] activations for one CFG image pair")
    return tensor[1:]


@contextmanager
def capture_features(unet: torch.nn.Module) -> Iterator[FeatureMaps]:
    """Observe CA, SA and block updates during one UNet forward.

    CA/SA are attention outputs before the enclosing residual addition. A block
    feature is FP32(output)-FP32(input), including SA, CA and feed-forward updates.
    Return conditional spatial/token means in FP32, shaped [1, channel].
    """
    captures: FeatureMaps = {}
    handles: list[RemovableHandle] = []
    inputs: LayerTensors = {}

    def remember_input(
        name: str, _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        inputs[name] = hidden.detach().clone()

    def record(
        representation: Representation,
        name: str,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        value = output.detach().float()
        if representation == "block":
            value = value - inputs.pop(name).float()
        value = conditional_branch(value)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {representation} activations at {name}")
        captures.setdefault(representation, {})[name] = value.mean(dim=1).cpu()

    try:
        for name, block in transformer_blocks(unet).items():
            handles.append(
                block.register_forward_pre_hook(partial(remember_input, name), with_kwargs=True)
            )
            for representation, module in (
                ("SA", block.attn1),
                ("CA", block.attn2),
                ("block", block),
            ):
                handles.append(module.register_forward_hook(partial(record, representation, name)))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def steered_tokens(conditional: torch.Tensor, delta: torch.Tensor, renorm: bool) -> torch.Tensor:
    """Add the channel offset to every token, optionally restoring per-token L2 norms.

    With renorm, each token keeps its pre-addition norm over the channel dim, so
    steering changes orientation only and injects no extra energy.
    """
    steered = conditional.float() + delta
    if renorm:
        original = conditional.float().norm(dim=-1, keepdim=True)
        steered = steered / steered.norm(dim=-1, keepdim=True).clamp(min=1e-8) * original
    return steered


@contextmanager
def apply_offsets(
    unet: torch.nn.Module,
    representation: Representation | None,
    deltas: LayerTensors,
    *,
    renorm: bool = False,
) -> Iterator[None]:
    """Add one channel offset to every conditional token at each selected location."""
    if not deltas:
        yield
        return
    if representation not in REPRESENTATIONS:
        raise ValueError("Injection requires a CA, SA or block representation")
    blocks = transformer_blocks(unet)
    if set(deltas) - set(blocks):
        raise ValueError("An injection names a missing transformer block")
    handles: list[RemovableHandle] = []

    def add(
        name: str, _module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        conditional = conditional_branch(output)
        delta = deltas[name].to(device=output.device, dtype=torch.float32)
        if delta.shape != (output.shape[-1],) or not torch.isfinite(delta).all():
            raise ValueError(f"Invalid channel offset at {name}")
        if not torch.any(delta):
            return output
        result = output.clone()
        result[1:] = steered_tokens(conditional, delta, renorm).to(output.dtype)
        return result

    try:
        for name in deltas:
            block = blocks[name]
            module = {"CA": block.attn2, "SA": block.attn1, "block": block}[representation]
            handles.append(module.register_forward_hook(partial(add, name)))
        yield
    finally:
        for handle in handles:
            handle.remove()
