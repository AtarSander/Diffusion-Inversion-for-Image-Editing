from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch

BRANCH_PAIR_MODE = "branch_pair"
PAIR_CONDITIONAL_ADAPTER_NAME = "text_branch"
PAIR_UNCONDITIONAL_ADAPTER_NAME = "null_branch"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_optional_path(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(to_absolute_path(path)).resolve()


def _lora_config_kwargs(lora_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    raw = (
        OmegaConf.to_container(lora_cfg, resolve=True)
        if isinstance(lora_cfg, DictConfig)
        else lora_cfg
    )
    ignored_keys = {
        "enabled",
        "checkpoint_path",
        "conditional_checkpoint_path",
        "unconditional_checkpoint_path",
        "adapter_name",
        "conditional_adapter_name",
        "unconditional_adapter_name",
        "scale",
        "mode",
        "active_steps",
        "active_fraction",
    }
    return {
        key: value for key, value in raw.items() if key not in ignored_keys and value is not None
    }


def _load_checkpoint(path: Path) -> Any:
    logger.info("Loading UNet LoRA checkpoint from {}", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _branch_pair_state_dicts(
    checkpoint_path: Path | None,
    conditional_checkpoint_path: Path | None,
    unconditional_checkpoint_path: Path | None,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    conditional_state = None
    unconditional_state = None

    if checkpoint_path is not None:
        state = _load_checkpoint(checkpoint_path)
        if not isinstance(state, dict):
            raise ValueError(f"Branch-pair checkpoint must be a dict: {checkpoint_path}")
        if "lora_state_dicts" in state:
            state = state["lora_state_dicts"]
        elif "adapters" in state:
            state = state["adapters"]
        if "conditional" not in state or "unconditional" not in state:
            raise ValueError(
                "Branch-pair checkpoint must contain 'conditional' and "
                f"'unconditional' adapter states: {checkpoint_path}"
            )
        conditional_state = state["conditional"]
        unconditional_state = state["unconditional"]

    if conditional_checkpoint_path is not None:
        conditional_state = _load_checkpoint(conditional_checkpoint_path)
    if unconditional_checkpoint_path is not None:
        unconditional_state = _load_checkpoint(unconditional_checkpoint_path)

    return conditional_state, unconditional_state


def _set_active_adapter(pipe, adapter_name: str) -> bool:
    switched = False
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
            switched = True
    if not switched:
        logger.warning("No LoRA adapter layers exposed set_adapter(); active adapter unchanged")
    return switched


def configure_unet_lora(pipe, lora_cfg: DictConfig | dict[str, Any] | None) -> bool:
    if not _cfg_get(lora_cfg, "enabled", False):
        return False

    from peft import LoraConfig, set_peft_model_state_dict

    mode = str(_cfg_get(lora_cfg, "mode", "single")).strip().lower()
    checkpoint_path = _resolve_optional_path(_cfg_get(lora_cfg, "checkpoint_path"))
    lora_config = LoraConfig(**_lora_config_kwargs(lora_cfg))

    if mode == BRANCH_PAIR_MODE:
        conditional_adapter_name = str(
            _cfg_get(lora_cfg, "conditional_adapter_name", PAIR_CONDITIONAL_ADAPTER_NAME)
        )
        unconditional_adapter_name = str(
            _cfg_get(lora_cfg, "unconditional_adapter_name", PAIR_UNCONDITIONAL_ADAPTER_NAME)
        )
        conditional_checkpoint_path = _resolve_optional_path(
            _cfg_get(lora_cfg, "conditional_checkpoint_path")
        )
        unconditional_checkpoint_path = _resolve_optional_path(
            _cfg_get(lora_cfg, "unconditional_checkpoint_path")
        )

        logger.info(
            "Adding UNet branch-pair LoRA adapters conditional='{}' unconditional='{}'",
            conditional_adapter_name,
            unconditional_adapter_name,
        )
        pipe.unet.add_adapter(lora_config, adapter_name=conditional_adapter_name)
        pipe.unet.add_adapter(lora_config, adapter_name=unconditional_adapter_name)

        conditional_state, unconditional_state = _branch_pair_state_dicts(
            checkpoint_path,
            conditional_checkpoint_path,
            unconditional_checkpoint_path,
        )
        if conditional_state is not None:
            set_peft_model_state_dict(
                pipe.unet,
                conditional_state,
                adapter_name=conditional_adapter_name,
            )
        if unconditional_state is not None:
            set_peft_model_state_dict(
                pipe.unet,
                unconditional_state,
                adapter_name=unconditional_adapter_name,
            )
        if conditional_state is None or unconditional_state is None:
            logger.info("Using randomly initialized weights for missing branch-pair adapters")

        _set_active_adapter(pipe, conditional_adapter_name)
        set_unet_lora_enabled(pipe, True)
        return True

    if mode not in {"single", "adapter"}:
        raise ValueError(f"Unsupported lora.mode={mode!r}; use 'single' or 'branch_pair'.")

    adapter_name = str(_cfg_get(lora_cfg, "adapter_name", "inversion"))
    logger.info("Adding UNet LoRA adapter '{}'", adapter_name)
    pipe.unet.add_adapter(lora_config, adapter_name=adapter_name)

    if checkpoint_path is not None:
        state_dict = _load_checkpoint(checkpoint_path)
        set_peft_model_state_dict(pipe.unet, state_dict, adapter_name=adapter_name)
    else:
        logger.info("Using randomly initialized LoRA adapter '{}'", adapter_name)

    scale = _cfg_get(lora_cfg, "scale", None)
    if scale is not None and hasattr(pipe, "set_adapters"):
        pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])

    set_unet_lora_enabled(pipe, True)

    return True


def get_lora_branch_adapter_names(
    lora_cfg: DictConfig | dict[str, Any] | None,
) -> tuple[str, str] | None:
    if not _cfg_get(lora_cfg, "enabled", False):
        return None
    mode = str(_cfg_get(lora_cfg, "mode", "single")).strip().lower()
    if mode != BRANCH_PAIR_MODE:
        return None
    return (
        str(_cfg_get(lora_cfg, "unconditional_adapter_name", PAIR_UNCONDITIONAL_ADAPTER_NAME)),
        str(_cfg_get(lora_cfg, "conditional_adapter_name", PAIR_CONDITIONAL_ADAPTER_NAME)),
    )


def set_unet_active_adapter(pipe, adapter_name: str) -> bool:
    return _set_active_adapter(pipe, adapter_name)


def set_unet_lora_enabled(pipe, enabled: bool) -> bool:
    toggled = False
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(enabled)
            toggled = True

    if not toggled:
        logger.warning("No LoRA adapter layers exposed enable_adapters(); toggle skipped")
    return toggled
