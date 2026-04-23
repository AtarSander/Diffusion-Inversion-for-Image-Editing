from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from loguru import logger
from omegaconf import DictConfig


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Resolve a torch dtype name from Hydra config."""
    if dtype_name is None:
        return None

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    return dtype


def make_pipe(cfg: DictConfig, device: str) -> StableDiffusionXLPipeline:
    """Create an SDXL pipeline configured for DDIM sampling."""
    logger.info("Loading SDXL pipeline: {}", cfg.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=resolve_torch_dtype(cfg.torch_dtype),
        variant=cfg.variant,
        use_safetensors=bool(cfg.use_safetensors),
    )
    if cfg.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    pipe.to(device)
    logger.success("SDXL pipeline loaded on {}", device)
    return pipe
