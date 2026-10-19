"""Read CA/SA/block updates and add offsets to the conditional branch."""

from collections.abc import Iterator
from contextlib import contextmanager
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


def conditional(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 2 * batch_size:
        raise ValueError("Expected BTC activations with unconditional then conditional batches")
    return tensor[batch_size:]


@contextmanager
def feature_hooks(
    unet: torch.nn.Module,
    *,
    batch_size: int = 1,
    captures: FeatureMaps | None = None,
    representation: Representation | None = None,
    deltas: LayerTensors | None = None,
) -> Iterator[None]:
    """Capture all three representations on one forward; inject one representation.

    CA/SA are attention outputs before the enclosing residual addition. A block
    feature is FP32(output)-FP32(input), while its offset is added at the output.
    Captures are spatial/token means in FP32.
    """
    if representation is not None and representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    blocks = transformer_blocks(unet)
    deltas = deltas or {}
    if deltas and representation is None:
        raise ValueError("Injection requires a representation")
    if set(deltas) - set(blocks):
        raise ValueError("An injection names a missing transformer block")
    handles: list[RemovableHandle] = []
    inputs: LayerTensors = {}

    def record(rep: Representation, name: str, tensor: torch.Tensor) -> None:
        if captures is None:
            return
        value = conditional(tensor, batch_size).detach().float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {rep} activations at {name}")
        captures.setdefault(rep, {})[name] = value.mean(dim=1).cpu()

    def inject(rep: Representation, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if representation != rep or name not in deltas:
            return tensor
        delta = deltas[name].to(device=tensor.device, dtype=torch.float32)
        if delta.shape != (tensor.shape[-1],) or not torch.isfinite(delta).all():
            raise ValueError(f"Invalid channel offset at {name}")
        if not torch.any(delta):
            return tensor
        result = tensor.clone()
        result[batch_size:] = (conditional(tensor, batch_size).float() + delta).to(tensor.dtype)
        return result

    try:
        for name, block in blocks.items():
            if captures is None and name not in deltas:
                continue

            def before(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                name: str = name,
            ) -> None:
                hidden = args[0] if args else kwargs["hidden_states"]
                inputs[name] = hidden.detach().clone()

            def after(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                output: torch.Tensor,
                *,
                name: str = name,
            ) -> torch.Tensor:
                record("block", name, output.detach().float() - inputs.pop(name).float())
                return inject("block", name, output)

            if captures is not None:
                handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(block.register_forward_hook(after))
            elif representation == "block":

                def add_block(
                    _module: torch.nn.Module,
                    _args: tuple[Any, ...],
                    output: torch.Tensor,
                    *,
                    name: str = name,
                ) -> torch.Tensor:
                    return inject("block", name, output)

                handles.append(block.register_forward_hook(add_block))
            for rep, module in (("SA", block.attn1), ("CA", block.attn2)):
                if captures is not None or representation == rep:

                    def hook(
                        _module: torch.nn.Module,
                        _args: tuple[Any, ...],
                        output: torch.Tensor,
                        *,
                        rep: Representation = rep,
                        name: str = name,
                    ) -> torch.Tensor:
                        record(rep, name, output)
                        return inject(rep, name, output)

                    handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
