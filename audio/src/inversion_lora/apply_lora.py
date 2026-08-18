# ABOUTME: Load a trained inversion-LoRA checkpoint into an AudioLDM2 UNet and return a toggle,
# ABOUTME: so the editing pipeline can enable the adapter for inversion and drop it for denoising.

import json
from pathlib import Path
from typing import Callable

import torch
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict


def attach_inversion_lora(unet, checkpoint_path: str | Path) -> Callable[[bool], None]:
    """Inject a trained inversion LoRA into `unet` and return a function to switch it on or off.

    The adapter is injected disabled. The caller decides where it applies: the shifted-denoiser
    objective trains eps_phi(x_{t-1}, t) to match eps_theta(x_t, t), which is the substitution
    DDIM *inversion* makes, so the adapter belongs on the inversion pass only and the reverse
    pass must run the frozen teacher.

    The LoRA configuration is read from the checkpoint's sidecar JSON rather than guessed, so
    rank and target modules always match what was trained.

    Args:
        unet: The AudioLDM2 UNet to modify in place.
        checkpoint_path: Adapter checkpoint written by `train.py`, with its `.json` sidecar.

    Returns:
        `set_enabled(bool)`, toggling every injected adapter layer.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    sidecar = checkpoint_path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} has no {sidecar.name}; the sidecar carries the LoRA config "
            "(rank, alpha, target modules) needed to rebuild the adapter."
        )

    meta = json.loads(sidecar.read_text())
    adapter_name = str(meta["adapter_name"])
    inject_adapter_in_model(LoraConfig(**meta["lora"]), unet, adapter_name=adapter_name)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    outcome = set_peft_model_state_dict(unet, state, adapter_name=adapter_name)
    unexpected = getattr(outcome, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"{checkpoint_path}: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    layers = [m for m in unet.modules() if m is not unet and hasattr(m, "enable_adapters")]
    if not layers:
        raise AttributeError("No injected LoRA layers exposed enable_adapters().")

    def set_enabled(enabled: bool) -> None:
        """Fold the adapter into the base weights, or take it back out.

        Merging rather than running the adapter as a side branch is what keeps this affordable:
        an unfused `full` adapter adds a branch to 1467 modules and cost 2.3x the no-LoRA time
        per edit, while merged weights cost nothing at all. The delta is the same either way.

        The order matters. PEFT's forward checks `disable_adapters` first and silently unmerges
        when it is set, so a merged-but-disabled layer quietly loses the adapter on its next
        forward. Adapters therefore stay enabled while merged, and `merged` alone decides
        whether the branch runs.
        """
        for layer in layers:
            if enabled:
                layer.enable_adapters(True)
                if not layer.merged:
                    layer.merge(adapter_names=[adapter_name])
            else:
                if layer.merged:
                    layer.unmerge()
                layer.enable_adapters(False)
        assert all(layer.merged == enabled for layer in layers), (
            f"{sum(layer.merged for layer in layers)}/{len(layers)} layers merged, "
            f"expected {'all' if enabled else 'none'}"
        )

    set_enabled(False)
    print(
        f"Loaded inversion LoRA {checkpoint_path.name} "
        f"(step {meta.get('global_step', '?')}, r={meta['lora'].get('r')}, "
        f"{len(layers)} layers, git {str(meta.get('git_sha', ''))[:8]})"
    )
    return set_enabled
