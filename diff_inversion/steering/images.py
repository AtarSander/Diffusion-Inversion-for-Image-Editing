"""SD1.5/SDXL loading, conditioning and deterministic image/latent conversion."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from diff_inversion.steering.lora import configure_unet_lora, set_unet_lora_enabled

from .types import Conditioning, Config

Pipeline = StableDiffusionPipeline | StableDiffusionXLPipeline


def load_pipeline(config: Config) -> Pipeline:
    model = config["model"]
    dtype = {"float16": torch.float16, "float32": torch.float32}[model["dtype"]]
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    for key in ("revision", "variant"):
        if model.get(key):
            kwargs[key] = model[key]
    family = model.get("family", "sdxl")
    if family not in ("sd15", "sdxl"):
        raise ValueError(f"Unsupported model family {family}")
    cls = StableDiffusionPipeline if family == "sd15" else StableDiffusionXLPipeline
    pipe = cls.from_pretrained(model["id"], **kwargs)
    if family == "sd15":
        # Compare every actual base target shape to the metadata-only preflight.
        import json
        evidence = json.loads((Path(os.environ["VISION_SD15_EVIDENCE_DIR"]) / "preflight.json").read_text())
        if model["id"] != evidence["model_id"] or model["revision"] != evidence["revision"]:
            raise ValueError("SD1.5 model identity differs from preflight")
        modules = dict(pipe.unet.named_modules())
        expected = evidence["tensors"]
        actual_targets = {n for n in modules if n.endswith(tuple("."+t for t in ("to_q","to_k","to_v","to_out.0")))}
        wanted = {k.removesuffix(".lora_A.weight") for k in expected if k.endswith(".lora_A.weight")}
        if actual_targets != wanted:
            raise ValueError("Runtime target names differ from metadata preflight")
        for name in wanted:
            a = expected[name+".lora_A.weight"]["shape"]
            b = expected[name+".lora_B.weight"]["shape"]
            if list(modules[name].weight.shape) != [b[0],a[1]]:
                raise ValueError(f"Runtime target shape mismatch: {name}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="leading")
    pipe.to(model["device"])
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.to(dtype=torch.float32).eval().requires_grad_(False)
    return pipe


def load_inversion_lora(pipe: Pipeline, lora: Config) -> None:
    """Load the MW adapter during extraction; fresh-noise generation uses only the base model."""
    configure_unet_lora(
        pipe,
        OmegaConf.create(
            {
                "checkpoint_path": lora["checkpoint_path"],
                "r": lora["r"],
                "lora_alpha": lora["lora_alpha"],
                "target_modules": lora["target_modules"],
                "enabled": True,
                "mode": "single",
                "scale": 1.0,
                "lora_dropout": 0.0,
                "bias": "none",
                "init_lora_weights": "gaussian",
            }
        ),
    )
    set_unet_lora_enabled(pipe, False)
    pipe.unet.eval().requires_grad_(False)


@torch.no_grad()
def conditioning(pipe: Pipeline, prompt: str, size: int) -> Conditioning:
    if isinstance(pipe, StableDiffusionPipeline):
        positive, negative = pipe.encode_prompt(
            prompt=prompt, negative_prompt="", device=pipe.device,
            do_classifier_free_guidance=True, num_images_per_prompt=1,
        )
        return {"prompt_embeds": positive, "negative_prompt_embeds": negative}
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
def encode_image(pipe: Pipeline, path: str | Path, size: int) -> torch.Tensor:
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
def decode_image(pipe: Pipeline, latent: torch.Tensor) -> Image.Image:
    latent = latent.to(device=pipe.device, dtype=torch.float32)
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
