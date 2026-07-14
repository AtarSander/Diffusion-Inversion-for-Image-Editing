from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from diff_inversion.modeling.cfg_temb import (
    CFG_TEMB_CONDITIONING_MODES,
    install_cfg_temb_conditioner,
    set_cfg_temb_enabled,
    split_cfg_temb_checkpoint_state,
)

BRANCH_PAIR_MODE = "branch_pair"
CFG_TEMB_MODE = "cfg_temb"
CFG_TEMB_PAIR_MODE = "cfg_temb_pair"
BRANCH_PAIR_MODES = frozenset({BRANCH_PAIR_MODE, CFG_TEMB_PAIR_MODE})
SINGLE_LORA_MODES = frozenset({"single", CFG_TEMB_MODE})
LORA_MODES = SINGLE_LORA_MODES | BRANCH_PAIR_MODES
SINGLE_ADAPTER_NAME = "inversion"
PAIR_CONDITIONAL_ADAPTER_NAME = "text_branch"
PAIR_UNCONDITIONAL_ADAPTER_NAME = "null_branch"


def _checkpoint_path(lora_cfg: DictConfig) -> Path:
    if not lora_cfg.checkpoint_path:
        raise ValueError("lora.checkpoint_path is required when lora.enabled=true.")
    path = Path(to_absolute_path(str(lora_cfg.checkpoint_path))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    return path


def _lora_config_kwargs(lora_cfg: DictConfig) -> dict[str, Any]:
    return {
        "r": int(lora_cfg.r),
        "lora_alpha": int(lora_cfg.lora_alpha),
        "lora_dropout": float(lora_cfg.lora_dropout),
        "bias": str(lora_cfg.bias),
        "init_lora_weights": lora_cfg.init_lora_weights,
        "target_modules": list(lora_cfg.target_modules),
    }


def _load_checkpoint(path: Path) -> Any:
    logger.info("Loading UNet LoRA checkpoint from {}", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _require_lora_state_dict(
    state: Any,
    *,
    label: str,
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    if (
        not isinstance(state, dict)
        or not state
        or any(not torch.is_tensor(value) for value in state.values())
    ):
        raise ValueError(f"Invalid {label} LoRA state dict in {checkpoint_path}.")
    return state


def _branch_pair_state_dicts(
    state: Any,
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not isinstance(state, dict) or state.get("branch_pair") is not True:
        raise ValueError(f"Not a branch-pair adapter checkpoint: {checkpoint_path}")
    adapters = state.get("adapters")
    if not isinstance(adapters, dict):
        raise ValueError(f"Branch-pair checkpoint is missing adapters: {checkpoint_path}")
    if "conditional" not in adapters or "unconditional" not in adapters:
        raise ValueError(
            "Branch-pair checkpoint must contain conditional and unconditional adapters: "
            f"{checkpoint_path}"
        )
    return (
        _require_lora_state_dict(
            adapters["conditional"],
            label="conditional",
            checkpoint_path=checkpoint_path,
        ),
        _require_lora_state_dict(
            adapters["unconditional"],
            label="unconditional",
            checkpoint_path=checkpoint_path,
        ),
    )


def _set_active_adapter(pipe, adapter_name: str) -> None:
    switched = False
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
            switched = True
    if not switched:
        raise AttributeError("No LoRA adapter layers expose set_adapter().")


def _configure_cfg_temb(
    pipe,
    *,
    lora_cfg: DictConfig,
    cfg_temb_state: dict[str, torch.Tensor],
    cfg_temb_config: dict[str, Any],
) -> None:
    conditioning_cfg = lora_cfg.cfg_temb_conditioning
    conditioner = install_cfg_temb_conditioner(
        pipe.unet,
        hidden_dim=int(conditioning_cfg.hidden_dim),
        log_mean=float(conditioning_cfg.log_mean),
        log_std=float(conditioning_cfg.log_std),
    )
    conditioner.validate_checkpoint_config(cfg_temb_config)
    conditioner.load_state_dict(cfg_temb_state)
    logger.info(
        "Enabled CFG timestep-embedding conditioner with hidden_dim={} temb_dim={} hooks={}",
        conditioner.hidden_dim,
        conditioner.temb_dim,
        conditioner.num_hooks,
    )


def configure_unet_lora(pipe, lora_cfg: DictConfig) -> bool:
    if not bool(lora_cfg.enabled):
        return False

    from peft import LoraConfig, set_peft_model_state_dict

    mode = str(lora_cfg.mode).strip().lower()
    if mode not in LORA_MODES:
        raise ValueError(f"Unsupported lora.mode={mode!r}; use one of {sorted(LORA_MODES)}.")

    checkpoint_path = _checkpoint_path(lora_cfg)
    checkpoint_state = _load_checkpoint(checkpoint_path)
    lora_state, cfg_temb_state, cfg_temb_config = split_cfg_temb_checkpoint_state(checkpoint_state)
    checkpoint_uses_cfg_temb = cfg_temb_state is not None
    mode_uses_cfg_temb = mode in CFG_TEMB_CONDITIONING_MODES
    if checkpoint_uses_cfg_temb != mode_uses_cfg_temb:
        if checkpoint_uses_cfg_temb:
            checkpoint_is_pair = (
                isinstance(lora_state, dict) and lora_state.get("branch_pair") is True
            )
            expected_mode = "cfg_temb_pair" if checkpoint_is_pair else "cfg_temb"
            raise ValueError(
                f"Checkpoint contains CFG-temb weights; use lora.mode={expected_mode}."
            )
        raise ValueError(f"lora.mode={mode} requires a CFG-temb checkpoint.")
    lora_config = LoraConfig(**_lora_config_kwargs(lora_cfg))

    if mode in BRANCH_PAIR_MODES:
        logger.info(
            "Adding UNet branch-pair LoRA adapters conditional='{}' unconditional='{}'",
            PAIR_CONDITIONAL_ADAPTER_NAME,
            PAIR_UNCONDITIONAL_ADAPTER_NAME,
        )
        pipe.unet.add_adapter(lora_config, adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME)
        pipe.unet.add_adapter(lora_config, adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME)

        conditional_state, unconditional_state = _branch_pair_state_dicts(
            lora_state,
            checkpoint_path,
        )
        set_peft_model_state_dict(
            pipe.unet,
            conditional_state,
            adapter_name=PAIR_CONDITIONAL_ADAPTER_NAME,
        )
        set_peft_model_state_dict(
            pipe.unet,
            unconditional_state,
            adapter_name=PAIR_UNCONDITIONAL_ADAPTER_NAME,
        )

    else:
        logger.info("Adding UNet LoRA adapter '{}'", SINGLE_ADAPTER_NAME)
        pipe.unet.add_adapter(lora_config, adapter_name=SINGLE_ADAPTER_NAME)
        state_dict = _require_lora_state_dict(
            lora_state,
            label="single-adapter",
            checkpoint_path=checkpoint_path,
        )
        set_peft_model_state_dict(pipe.unet, state_dict, adapter_name=SINGLE_ADAPTER_NAME)

    if mode_uses_cfg_temb:
        assert cfg_temb_state is not None
        assert cfg_temb_config is not None
        _configure_cfg_temb(
            pipe,
            lora_cfg=lora_cfg,
            cfg_temb_state=cfg_temb_state,
            cfg_temb_config=cfg_temb_config,
        )

    if mode in BRANCH_PAIR_MODES:
        _set_active_adapter(pipe, PAIR_CONDITIONAL_ADAPTER_NAME)
    elif lora_cfg.scale is not None:
        pipe.set_adapters(
            [SINGLE_ADAPTER_NAME],
            adapter_weights=[float(lora_cfg.scale)],
        )

    set_unet_lora_enabled(pipe, True)

    return True


def get_lora_branch_adapter_names(
    lora_cfg: DictConfig,
) -> tuple[str, str] | None:
    if not bool(lora_cfg.enabled):
        return None
    mode = str(lora_cfg.mode).strip().lower()
    if mode not in BRANCH_PAIR_MODES:
        return None
    return (
        PAIR_UNCONDITIONAL_ADAPTER_NAME,
        PAIR_CONDITIONAL_ADAPTER_NAME,
    )


def set_unet_lora_enabled(pipe, enabled: bool) -> bool:
    toggled = False
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(enabled)
            toggled = True

    if not toggled:
        raise AttributeError("No LoRA adapter layers expose enable_adapters().")
    set_cfg_temb_enabled(pipe.unet, enabled)
    return toggled
