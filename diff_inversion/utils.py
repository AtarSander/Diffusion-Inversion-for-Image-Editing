from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Resolve a torch dtype name from Hydra config."""
    if dtype_name is None:
        return None

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    return dtype


def _pipeline_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "pipeline", default="sdxl")).lower()


def _from_pretrained_kwargs(cfg: DictConfig) -> dict:
    kwargs = {
        "torch_dtype": resolve_torch_dtype(OmegaConf.select(cfg, "torch_dtype", default=None)),
    }
    variant = OmegaConf.select(cfg, "variant", default=None)
    if variant is not None:
        kwargs["variant"] = variant
    revision = OmegaConf.select(cfg, "revision", default=None)
    if revision is not None:
        kwargs["revision"] = revision
    use_safetensors = OmegaConf.select(cfg, "use_safetensors", default=None)
    if use_safetensors is not None:
        kwargs["use_safetensors"] = bool(use_safetensors)
    return kwargs


def make_pipe(cfg: DictConfig, device: str) -> StableDiffusionPipeline | StableDiffusionXLPipeline:
    """Create a Stable Diffusion pipeline configured for DDIM sampling."""
    pipeline_name = _pipeline_name(cfg)
    pipeline_cls = {
        "sdxl": StableDiffusionXLPipeline,
        "stable-diffusion-xl": StableDiffusionXLPipeline,
        "sd15": StableDiffusionPipeline,
        "sd1.5": StableDiffusionPipeline,
        "stable-diffusion": StableDiffusionPipeline,
    }.get(pipeline_name)
    if pipeline_cls is None:
        raise ValueError(
            f"Unsupported model.pipeline={pipeline_name!r}. Expected one of: sdxl, sd15."
        )

    logger.info("Loading {} pipeline: {}", pipeline_name, cfg.model_id)
    pipe = pipeline_cls.from_pretrained(cfg.model_id, **_from_pretrained_kwargs(cfg))
    if cfg.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    pipe.to(device)
    logger.success("{} pipeline loaded on {}", pipeline_name, device)
    return pipe
