"""Reconstruct SDXL images from saved inverted DDIM noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
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
    predict_noise_sdxl,
)


@torch.no_grad()
def reconstruct_from_noise_sdxl(
    pipe,
    inverted_noise: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    model_cfg,
) -> tuple[torch.Tensor, list[int]]:
    """Run DDIM denoising from an inverted noise latent back to an image latent."""
    if inverted_noise.ndim == 3:
        inverted_noise = inverted_noise.unsqueeze(0)
    if inverted_noise.ndim != 4 or inverted_noise.shape[0] != 1:
        raise ValueError(
            "Expected inverted noise with shape [1,C,H,W] or [C,H,W], "
            f"got {tuple(inverted_noise.shape)}"
        )

    latents = inverted_noise.to(device=pipe.device, dtype=pipe.unet.dtype)
    cond = encode_prompt_sdxl(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=model_cfg.height,
        width=model_cfg.width,
    )

    pipe.scheduler.set_timesteps(model_cfg.num_inference_steps, device=pipe.device)
    timestep_values = []
    for timestep in tqdm(pipe.scheduler.timesteps, desc="Reconstructing", leave=False):
        timestep_values.append(
            int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
        )
        noise_pred = predict_noise_sdxl(
            pipe=pipe,
            latents=latents,
            timestep=timestep,
            cond=cond,
            guidance_scale=model_cfg.guidance_scale,
        )
        latents = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            return_dict=True,
        ).prev_sample

    return latents.detach().cpu(), timestep_values


@torch.no_grad()
def reconstruct_sample(
    pipe,
    sample_dir: Path,
    model_cfg,
    negative_prompt: str,
    output_name: str,
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
    prompt = _read_json(prompt_path)["prompt"]

    inverted_noise = _load_tensor(inverted_noise_path)
    reconstructed_latent, timestep_values = reconstruct_from_noise_sdxl(
        pipe=pipe,
        inverted_noise=inverted_noise,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_cfg=model_cfg,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/sdxl_trajectories"),
        help="Directory containing sample_* with inverted_noise.pt.",
    )
    parser.add_argument(
        "--output-name",
        default="reconstructed.png",
        help="Image filename to write inside each sample directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_run_config(args.input_dir)
    samples = _sample_dirs(args.input_dir)
    if not samples:
        raise FileNotFoundError(f"No sample directories found in {args.input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(cfg.model, device)
    negative_prompt = str(cfg.negative_prompt)
    for sample_dir in tqdm(samples, desc="Running SDXL reconstruction"):
        reconstruct_sample(
            pipe=pipe,
            sample_dir=sample_dir,
            model_cfg=cfg.model,
            negative_prompt=negative_prompt,
            output_name=args.output_name,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
