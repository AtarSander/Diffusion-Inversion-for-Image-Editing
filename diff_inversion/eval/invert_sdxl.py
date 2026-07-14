"""Run baseline DDIM inversion for saved SDXL trajectory samples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import DDIMInverseScheduler
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    encode_prompt_sdxl,
    make_pipe,
)
from diff_inversion.eval.lora import (
    configure_unet_lora,
    get_lora_branch_adapter_names,
    set_unet_lora_enabled,
)
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


def _load_final_latent(sample_dir: Path) -> torch.Tensor:
    latent_paths = sorted((sample_dir / "latents").glob("x_*.pt"))
    if latent_paths:
        return _load_tensor(latent_paths[-1])

    trajectory_path = sample_dir / "latents" / "trajectory.pt"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"No generated latents found in {sample_dir / 'latents'}")

    trajectory = _load_tensor(trajectory_path)
    if trajectory.ndim == 5 and trajectory.shape[1] == 1:
        return trajectory[-1]
    if trajectory.ndim == 4:
        return trajectory[-1].unsqueeze(0)

    raise ValueError(
        "Expected stacked trajectory with shape [T,1,C,H,W] or [T,C,H,W], "
        f"got {tuple(trajectory.shape)}"
    )


def _lora_active_steps(lora_cfg: DictConfig, total_steps: int) -> int | None:
    if not bool(lora_cfg.enabled):
        return None

    if lora_cfg.active_steps is not None:
        return max(0, min(total_steps, int(lora_cfg.active_steps)))

    if lora_cfg.active_fraction is None:
        return None

    fraction = float(lora_cfg.active_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"lora.active_fraction must be in [0, 1], got {fraction}")
    return max(0, min(total_steps, math.ceil(total_steps * fraction)))


@torch.no_grad()
def invert_sample(
    pipe,
    sample_dir: Path,
    model_cfg,
    negative_prompt: str,
    lora_cfg,
    lora_loaded: bool,
    overwrite: bool,
    save_inversion_latents: bool,
    save_inversion_pred_noises: bool,
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

    final_latent = _load_final_latent(sample_dir).to(
        device=pipe.device,
        dtype=pipe.unet.dtype,
    )

    original_scheduler = pipe.scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
    pipe.scheduler = inverse_scheduler
    active_lora_steps = None
    try:
        cond = encode_prompt_sdxl(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=model_cfg.height,
            width=model_cfg.width,
        )
        inverse_scheduler.set_timesteps(model_cfg.num_inference_steps, device=pipe.device)
        active_lora_steps = (
            _lora_active_steps(lora_cfg, total_steps=len(inverse_scheduler.timesteps))
            if lora_loaded
            else None
        )
        if active_lora_steps is not None:
            logger.info(
                "Using LoRA for the first {}/{} inversion steps in {}",
                active_lora_steps,
                len(inverse_scheduler.timesteps),
                sample_dir.name,
            )
        lora_branch_adapter_names = (
            get_lora_branch_adapter_names(lora_cfg) if lora_loaded else None
        )

        latents = final_latent
        inversion_trajectory = [latents.detach().cpu()] if save_inversion_latents else []
        inversion_pred_noises = []
        inversion_timesteps = [
            int(inverse_scheduler.timesteps[0].item())
            if hasattr(inverse_scheduler.timesteps[0], "item")
            else int(inverse_scheduler.timesteps[0])
        ]

        current_lora_enabled = None
        for step_idx, timestep in enumerate(
            tqdm(inverse_scheduler.timesteps, desc=f"Inverting {sample_dir.name}", leave=False)
        ):
            if active_lora_steps is not None:
                should_enable_lora = step_idx < active_lora_steps
                if should_enable_lora != current_lora_enabled:
                    set_unet_lora_enabled(pipe, should_enable_lora)
                    current_lora_enabled = should_enable_lora

            noise_pred = predict_noise_sdxl(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                cond=cond,
                guidance_scale=model_cfg.guidance_scale,
                lora_branch_adapter_names=lora_branch_adapter_names,
            )
            latents = inverse_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                return_dict=True,
            ).prev_sample
            if save_inversion_latents:
                inversion_trajectory.append(latents.detach().cpu())
            if save_inversion_pred_noises:
                inversion_pred_noises.append(noise_pred.detach().cpu())
            inversion_timesteps.append(
                int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            )
    finally:
        if lora_loaded and active_lora_steps is not None:
            set_unet_lora_enabled(pipe, True)
        pipe.scheduler = original_scheduler

    torch.save(latents.detach().cpu(), inverted_noise_path)
    if save_inversion_latents:
        inversion_latents_dir.mkdir(parents=True, exist_ok=True)
        for idx, latent in enumerate(inversion_trajectory):
            torch.save(latent, inversion_latents_dir / f"x_inv_{idx:03d}.pt")
    if save_inversion_pred_noises:
        inversion_pred_noises_dir.mkdir(parents=True, exist_ok=True)
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
                "inversion_trajectory_saved": bool(save_inversion_latents),
                "inversion_pred_noises_saved": bool(save_inversion_pred_noises),
                "inversion_lora_enabled": bool(lora_loaded),
                "inversion_lora_active_steps": active_lora_steps,
            }
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.success("Saved inversion for {}", sample_dir)


@hydra.main(config_path="../../config", config_name="eval/invert_sdxl", version_base=None)
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
    lora_loaded = configure_unet_lora(pipe, cfg.lora)
    negative_prompt = str(run_cfg.negative_prompt)
    for sample_dir in tqdm(samples, desc="Running DDIM inversion"):
        invert_sample(
            pipe=pipe,
            sample_dir=sample_dir,
            model_cfg=run_cfg.model,
            negative_prompt=negative_prompt,
            lora_cfg=cfg.lora,
            lora_loaded=lora_loaded,
            overwrite=bool(cfg.overwrite),
            save_inversion_latents=bool(cfg.save_inversion_latents),
            save_inversion_pred_noises=bool(cfg.save_inversion_pred_noises),
        )


if __name__ == "__main__":
    main()
