from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch


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
        "adapter_name",
        "scale",
        "mode",
        "active_steps",
        "active_fraction",
    }
    return {
        key: value for key, value in raw.items() if key not in ignored_keys and value is not None
    }


def configure_unet_lora(pipe, lora_cfg: DictConfig | dict[str, Any] | None) -> bool:
    if not _cfg_get(lora_cfg, "enabled", False):
        return False

    from peft import LoraConfig, set_peft_model_state_dict

    adapter_name = str(_cfg_get(lora_cfg, "adapter_name", "inversion"))
    checkpoint_path = _resolve_optional_path(_cfg_get(lora_cfg, "checkpoint_path"))
    lora_config = LoraConfig(**_lora_config_kwargs(lora_cfg))

    logger.info("Adding UNet LoRA adapter '{}'", adapter_name)
    pipe.unet.add_adapter(lora_config, adapter_name=adapter_name)

    if checkpoint_path is not None:
        logger.info("Loading UNet LoRA checkpoint from {}", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(pipe.unet, state_dict, adapter_name=adapter_name)
    else:
        logger.info("Using randomly initialized LoRA adapter '{}'", adapter_name)

    scale = _cfg_get(lora_cfg, "scale", None)
    if scale is not None and hasattr(pipe, "set_adapters"):
        pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])

    set_unet_lora_enabled(pipe, True)

    return True


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
