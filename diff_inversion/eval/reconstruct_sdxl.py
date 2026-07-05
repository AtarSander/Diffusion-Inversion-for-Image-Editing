"""Reconstruct SDXL images from saved inverted DDIM noise."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    decode_latent_to_pil,
    encode_prompt_sdxl,
    make_pipe,
)
from diff_inversion.eval.invert_sdxl import (
    _load_run_config,
    _load_tensor,
    _read_json,
    _sample_dirs,
)
from diff_inversion.eval.lora import configure_unet_lora, get_lora_branch_adapter_names
from diff_inversion.modeling.sdxl_sampling import reconstruct_latent_sdxl


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


@torch.no_grad()
def reconstruct_from_noise_sdxl(
    pipe,
    inverted_noise: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    model_cfg,
    lora_branch_adapter_names: tuple[str, str] | None = None,
    guidance_scale: float | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Run DDIM denoising from an inverted noise latent back to an image latent."""
    if inverted_noise.ndim == 3:
        inverted_noise = inverted_noise.unsqueeze(0)
    if inverted_noise.ndim != 4 or inverted_noise.shape[0] != 1:
        raise ValueError(
            "Expected inverted noise with shape [1,C,H,W] or [C,H,W], "
            f"got {tuple(inverted_noise.shape)}"
        )

    cond = encode_prompt_sdxl(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=model_cfg.height,
        width=model_cfg.width,
    )
    if guidance_scale is None:
        guidance_scale = float(model_cfg.guidance_scale)
    return reconstruct_latent_sdxl(
        pipe=pipe,
        noise_latent=inverted_noise,
        cond=cond,
        num_inference_steps=model_cfg.num_inference_steps,
        guidance_scale=guidance_scale,
        lora_branch_adapter_names=lora_branch_adapter_names,
        progress_desc="Reconstructing",
    )


@torch.no_grad()
def reconstruct_sample(
    pipe,
    sample_dir: Path,
    model_cfg,
    negative_prompt: str,
    output_name: str,
    prompt_field: str,
    lora_branch_adapter_names: tuple[str, str] | None,
    guidance_scale: float | None,
    overwrite: bool,
) -> None:
    inverted_noise_path = sample_dir / "inverted_noise.pt"
    output_path = sample_dir / output_name
    timesteps_path = sample_dir / "reconstruction_timesteps.json"

    if output_path.exists() and not overwrite:
        logger.info("Skipping existing reconstruction: {}", sample_dir)
        return
    if not inverted_noise_path.exists():
        logger.warning("Skipping {} because inverted_noise.pt is missing", sample_dir)
        return

    prompt_path = sample_dir / "prompt.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt metadata not found: {prompt_path}")
    prompt_record = _read_json(prompt_path)
    prompt = prompt_record.get(prompt_field) or prompt_record["prompt"]

    inverted_noise = _load_tensor(inverted_noise_path)
    reconstructed_latent, timestep_values = reconstruct_from_noise_sdxl(
        pipe=pipe,
        inverted_noise=inverted_noise,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_cfg=model_cfg,
        lora_branch_adapter_names=lora_branch_adapter_names,
        guidance_scale=guidance_scale,
    )
    image = decode_latent_to_pil(pipe, reconstructed_latent.to(device=pipe.device))
    image.save(output_path)

    with timesteps_path.open("w", encoding="utf-8") as f:
        json.dump(timestep_values, f, indent=2)

    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        meta.update(
            {
                "reconstructed_image": output_path.name,
                "reconstruction_timesteps": timesteps_path.name,
            }
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.success("Saved reconstruction for {}", sample_dir)


@hydra.main(config_path="../../config", config_name="eval/reconstruct_sdxl", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Reconstruction config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = _resolve_path(cfg.input_dir)
    run_cfg = _load_run_config(input_dir)
    samples = _sample_dirs(input_dir)
    if not samples:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if run_cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(run_cfg.model, device)
    lora_loaded = configure_unet_lora(pipe, cfg.lora)
    lora_branch_adapter_names = get_lora_branch_adapter_names(cfg.lora) if lora_loaded else None
    negative_prompt = str(run_cfg.negative_prompt)
    guidance_scale = OmegaConf.select(cfg, "guidance_scale", default=None)
    guidance_scale = None if guidance_scale is None else float(guidance_scale)
    for sample_dir in tqdm(samples, desc="Running SDXL reconstruction"):
        reconstruct_sample(
            pipe=pipe,
            sample_dir=sample_dir,
            model_cfg=run_cfg.model,
            negative_prompt=negative_prompt,
            output_name=str(cfg.output_name),
            prompt_field=str(cfg.prompt_field),
            lora_branch_adapter_names=lora_branch_adapter_names,
            guidance_scale=guidance_scale,
            overwrite=bool(cfg.overwrite),
        )


if __name__ == "__main__":
    main()
