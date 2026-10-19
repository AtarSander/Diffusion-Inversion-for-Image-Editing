"""Read CA/SA/block updates and add offsets to the conditional branch."""

from contextlib import contextmanager

import torch
from diffusers.models.attention import BasicTransformerBlock

REPRESENTATIONS = ("CA", "SA", "block")


def transformer_blocks(unet):
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


def conditional(tensor, batch_size):
    if tensor.ndim != 3 or tensor.shape[0] != 2 * batch_size:
        raise ValueError("Expected BTC activations with unconditional then conditional batches")
    return tensor[batch_size:]


@contextmanager
def feature_hooks(
    unet, *, batch_size=1, captures=None, moments=None, representation=None, deltas=None
):
    """Capture all three representations on one forward; inject one representation.

    CA/SA are attention outputs before the enclosing residual addition. A block
    feature is FP32(output)-FP32(input), while its offset is added at the output.
    Captures are spatial/token means; moments are full-map second moments.
    """
    if representation is not None and representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    blocks = transformer_blocks(unet)
    deltas = deltas or {}
    if deltas and representation is None:
        raise ValueError("Injection requires a representation")
    if set(deltas) - set(blocks):
        raise ValueError("An injection names a missing transformer block")
    handles, inputs = [], {}

    def record(rep, name, tensor):
        if captures is None and moments is None:
            return
        value = conditional(tensor, batch_size).detach().float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {rep} activations at {name}")
        if captures is not None:
            captures.setdefault(rep, {})[name] = value.mean(dim=1).cpu()
        if moments is not None:
            moments.setdefault(rep, {})[name] = (
                value.square().mean(dim=(1, 2), dtype=torch.float64).cpu()
            )

    def inject(rep, name, tensor):
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
            if captures is None and moments is None and name not in deltas:
                continue

            def before(_module, args, kwargs, *, name=name):
                hidden = args[0] if args else kwargs["hidden_states"]
                inputs[name] = hidden.detach().clone()

            def after(_module, _args, output, *, name=name):
                record("block", name, output.detach().float() - inputs.pop(name).float())
                return inject("block", name, output)

            if captures is not None or moments is not None:
                handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(block.register_forward_hook(after))
            elif representation == "block":

                def add_block(_module, _args, output, *, name=name):
                    return inject("block", name, output)

                handles.append(block.register_forward_hook(add_block))
            for rep, module in (("SA", block.attn1), ("CA", block.attn2)):
                if captures is not None or moments is not None or representation == rep:

                    def hook(_module, _args, output, *, rep=rep, name=name):
                        record(rep, name, output)
                        return inject(rep, name, output)

                    handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
