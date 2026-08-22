"""LoRA checkpoint loading and adapter switching for SD1.5 editors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

PAIR_CONDITIONAL_ADAPTER_NAME = "text_branch"
PAIR_UNCONDITIONAL_ADAPTER_NAME = "null_branch"


def _load_checkpoint(path: str | Path) -> tuple[Path, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint_path, state


def _lora_config(rank: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )


def _branch_pair_states(
    state: Any,
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not isinstance(state, dict):
        raise ValueError(f"Not a branch-pair LoRA checkpoint: {checkpoint_path}")
    if state.get("branch_pair") is True:
        adapters = state.get("adapters")
    else:
        adapters = state.get("lora_state_dicts")
    if not isinstance(adapters, dict):
        raise ValueError(f"Not a branch-pair LoRA checkpoint: {checkpoint_path}")
    conditional = adapters.get("conditional")
    unconditional = adapters.get("unconditional")
    if not isinstance(conditional, dict) or not isinstance(unconditional, dict):
        raise ValueError(
            f"Branch-pair checkpoint lacks conditional/unconditional adapters: {checkpoint_path}"
        )
    return conditional, unconditional


def set_active_lora_adapter(unet, adapter_name: str, scale: float = 1.0) -> None:
    """Activate exactly one injected adapter for the next UNet forward."""
    if hasattr(unet, "set_adapters"):
        unet.set_adapters([adapter_name], weights=[float(scale)])
        return

    switched = False
    for module in unet.modules():
        if module is unet:
            continue
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
            switched = True
    if not switched:
        raise AttributeError("No injected LoRA layers expose adapter switching.")


def set_lora_enabled(unet, enabled: bool) -> None:
    toggled = False
    for module in unet.modules():
        if module is unet:
            continue
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(bool(enabled))
            toggled = True
    if not toggled:
        raise AttributeError("No injected LoRA layers expose enable_adapters().")


def load_branch_pair_lora(
    unet,
    checkpoint_path: str | Path,
    *,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    scale: float = 1.0,
) -> tuple[str, tuple[str, str]]:
    """Load conditional and empty-prompt LoRAs from a branch-pair checkpoint."""
    from peft import inject_adapter_in_model, set_peft_model_state_dict

    resolved_path, state = _load_checkpoint(checkpoint_path)
    conditional_state, unconditional_state = _branch_pair_states(state, resolved_path)
    config = _lora_config(rank, lora_alpha, lora_dropout)
    inject_adapter_in_model(config, unet, adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME)
    inject_adapter_in_model(config, unet, adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME)
    set_peft_model_state_dict(
        unet,
        conditional_state,
        adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME,
    )
    set_peft_model_state_dict(
        unet,
        unconditional_state,
        adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME,
    )
    set_active_lora_adapter(unet, PAIR_CONDITIONAL_ADAPTER_NAME, scale)
    set_lora_enabled(unet, False)
    return str(resolved_path), (
        PAIR_UNCONDITIONAL_ADAPTER_NAME,
        PAIR_CONDITIONAL_ADAPTER_NAME,
    )
