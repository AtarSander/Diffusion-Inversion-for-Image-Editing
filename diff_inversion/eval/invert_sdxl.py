"""Run baseline DDIM inversion for saved SDXL trajectory samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diffusers import DDIMInverseScheduler
from loguru import logger
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    encode_prompt_sdxl,
    make_pipe,
)


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(tensor)!r}")
    return tensor


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_config(input_dir: Path):
    run_config_path = input_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"Run config not found: {run_config_path}. "
            "Run `make generate-baseline-samples` first or point --input-dir at its output."
        )
    return OmegaConf.create(_read_json(run_config_path))


def _sample_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("sample_*") if path.is_dir())


def _final_latent_path(sample_dir: Path) -> Path:
    latent_paths = sorted((sample_dir / "latents").glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No generated latents found in {sample_dir / 'latents'}")
    return latent_paths[-1]


@torch.no_grad()
def predict_noise_sdxl(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: dict[str, torch.Tensor],
    guidance_scale: float,
) -> torch.Tensor:
    """Predict CFG-combined SDXL noise for one latent batch."""
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    encoder_hidden_states = torch.cat(
        [cond["negative_prompt_embeds"], cond["prompt_embeds"]],
        dim=0,
    )
    text_embeds = torch.cat(
        [cond["negative_pooled_prompt_embeds"], cond["pooled_prompt_embeds"]],
        dim=0,
    )
    time_ids = cond["add_time_ids"].repeat(2, 1)

    noise_pred = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        },
        return_dict=False,
    )[0]

    noise_uncond, noise_text = noise_pred.chunk(2)
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)


@torch.no_grad()
def invert_sample(
    pipe,
    sample_dir: Path,
    model_cfg,
    negative_prompt: str,
    overwrite: bool,
) -> None:
    inverted_noise_path = sample_dir / "inverted_noise.pt"
    inversion_latents_dir = sample_dir / "inversion_latents"
    inversion_pred_noises_dir = sample_dir / "inversion_pred_noises"
    inversion_timesteps_path = sample_dir / "inversion_timesteps.json"

    if inverted_noise_path.exists() and not overwrite:
        logger.info("Skipping existing inversion: {}", sample_dir)
        return

    prompt_path = sample_dir / "prompt.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt metadata not found: {prompt_path}")
    prompt = _read_json(prompt_path)["prompt"]

    final_latent = _load_tensor(_final_latent_path(sample_dir)).to(
        device=pipe.device,
        dtype=pipe.unet.dtype,
    )

    original_scheduler = pipe.scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
    pipe.scheduler = inverse_scheduler
    try:
        cond = encode_prompt_sdxl(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=model_cfg.height,
            width=model_cfg.width,
        )
        inverse_scheduler.set_timesteps(model_cfg.num_inference_steps, device=pipe.device)

        latents = final_latent
        inversion_trajectory = [latents.detach().cpu()]
        inversion_pred_noises = []
        inversion_timesteps = [
            int(inverse_scheduler.timesteps[0].item())
            if hasattr(inverse_scheduler.timesteps[0], "item")
            else int(inverse_scheduler.timesteps[0])
        ]

        for timestep in tqdm(inverse_scheduler.timesteps, desc=f"Inverting {sample_dir.name}", leave=False):
            noise_pred = predict_noise_sdxl(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                cond=cond,
                guidance_scale=model_cfg.guidance_scale,
            )
            latents = inverse_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                return_dict=True,
            ).prev_sample
            inversion_trajectory.append(latents.detach().cpu())
            inversion_pred_noises.append(noise_pred.detach().cpu())
            inversion_timesteps.append(int(timestep.item()) if hasattr(timestep, "item") else int(timestep))
    finally:
        pipe.scheduler = original_scheduler

    inversion_latents_dir.mkdir(parents=True, exist_ok=True)
    inversion_pred_noises_dir.mkdir(parents=True, exist_ok=True)

    torch.save(latents.detach().cpu(), inverted_noise_path)
    for idx, latent in enumerate(inversion_trajectory):
        torch.save(latent, inversion_latents_dir / f"x_inv_{idx:03d}.pt")
    for idx, noise in enumerate(inversion_pred_noises):
        torch.save(noise, inversion_pred_noises_dir / f"noise_inv_{idx:03d}.pt")
    with inversion_timesteps_path.open("w", encoding="utf-8") as f:
        json.dump(inversion_timesteps, f, indent=2)

    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        meta.update(
            {
                "inverted_noise": inverted_noise_path.name,
                "inversion_trajectory_length": len(inversion_trajectory),
                "inversion_pred_noises_length": len(inversion_pred_noises),
            }
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.success("Saved inversion for {}", sample_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/sdxl_trajectories"),
        help="Directory containing sample_* generated by generate_sdxl_samples.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing inversions.")
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
    for sample_dir in tqdm(samples, desc="Running DDIM inversion"):
        invert_sample(
            pipe=pipe,
            sample_dir=sample_dir,
            model_cfg=cfg.model,
            negative_prompt=negative_prompt,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
