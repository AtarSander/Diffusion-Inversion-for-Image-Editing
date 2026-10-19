"""SDXL loading, text conditioning and deterministic image/latent conversion."""

from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from diff_inversion.eval.lora import configure_unet_lora, set_unet_lora_enabled

from .types import Conditioning, Config


def load_pipeline(config: Config) -> StableDiffusionXLPipeline:
    model = config["model"]
    dtype = {"float16": torch.float16, "float32": torch.float32}[model["dtype"]]
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    for key in ("revision", "variant"):
        if model.get(key):
            kwargs[key] = model[key]
    pipe = StableDiffusionXLPipeline.from_pretrained(model["id"], **kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="leading")
    pipe.to(model["device"])
    if config.get("lora"):
        configure_unet_lora(
            pipe,
            OmegaConf.create(
                {
                    "enabled": True,
                    "mode": "single",
                    "scale": 1.0,
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "init_lora_weights": "gaussian",
                    **config["lora"],
                }
            ),
        )
        set_unet_lora_enabled(pipe, False)
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.to(dtype=torch.float32).eval().requires_grad_(False)
    return pipe


@torch.no_grad()
def conditioning(pipe: StableDiffusionXLPipeline, prompt: str, size: int) -> Conditioning:
    positive, negative, pooled, negative_pooled = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt="",
        negative_prompt_2="",
        device=pipe.device,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
    )
    time_ids = pipe._get_add_time_ids(
        original_size=(size, size),
        crops_coords_top_left=(0, 0),
        target_size=(size, size),
        dtype=positive.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(pipe.device)
    return {
        "prompt_embeds": positive,
        "negative_prompt_embeds": negative,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": negative_pooled,
        "add_time_ids": time_ids,
    }


@torch.no_grad()
def encode_image(pipe: StableDiffusionXLPipeline, path: str | Path, size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1)[None].to(pipe.device).div(127.5).sub(1)
    latent = pipe.vae.encode(pixels).latent_dist.mode() * pipe.vae.config.scaling_factor
    return latent.to(dtype=pipe.unet.dtype)


@torch.no_grad()
def decode_image(pipe: StableDiffusionXLPipeline, latent: torch.Tensor) -> Image.Image:
    latent = latent.to(device=pipe.device, dtype=torch.float32)
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
