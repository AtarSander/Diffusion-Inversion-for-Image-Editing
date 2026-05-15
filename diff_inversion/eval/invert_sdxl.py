"""Run baseline DDIM inversion for saved SDXL trajectory samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diffusers import DDIMInverseScheduler
import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    encode_prompt_sdxl,
    make_pipe,
)
from diff_inversion.eval.lora import configure_unet_lora
from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl


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
            "Run `make generate-baseline-samples` first or set input_dir to its output."
        )
    return OmegaConf.create(_read_json(run_config_path))


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _sample_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("sample_*") if path.is_dir())


def _final_latent_path(sample_dir: Path) -> Path:
    latent_paths = sorted((sample_dir / "latents").glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No generated latents found in {sample_dir / 'latents'}")
    return latent_paths[-1]


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

        for timestep in tqdm(
            inverse_scheduler.timesteps, desc=f"Inverting {sample_dir.name}", leave=False
        ):
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
            inversion_timesteps.append(
                int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            )
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


@hydra.main(config_path="../../config/eval", config_name="invert_sdxl", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Inversion config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = _resolve_path(cfg.input_dir)
    run_cfg = _load_run_config(input_dir)
    samples = _sample_dirs(input_dir)
    if not samples:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if run_cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(run_cfg.model, device)
    configure_unet_lora(pipe, cfg.lora)
    negative_prompt = str(run_cfg.negative_prompt)
    for sample_dir in tqdm(samples, desc="Running DDIM inversion"):
        invert_sample(
            pipe=pipe,
            sample_dir=sample_dir,
            model_cfg=run_cfg.model,
            negative_prompt=negative_prompt,
            overwrite=bool(cfg.overwrite),
        )


if __name__ == "__main__":
    main()
