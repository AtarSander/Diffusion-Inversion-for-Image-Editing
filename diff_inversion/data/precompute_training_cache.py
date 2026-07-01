"""Add cached SDXL inversion-training tensors to existing trajectory samples."""

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diff_inversion.data.generate_sdxl_samples import (
    encode_prompt_sdxl,
    has_sdxl_conditioning,
    save_training_cache,
)
from diff_inversion.utils import make_pipe


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cache_paths(sample_dir: Path, cfg: DictConfig) -> tuple[Path, Path, Path]:
    conditioning_path = sample_dir / str(
        OmegaConf.select(cfg, "conditioning_file_name", default="conditioning.pt")
    )
    targets_dir = sample_dir / str(OmegaConf.select(cfg, "targets_dir_name", default="targets"))
    target_eps_path = (
        targets_dir / str(OmegaConf.select(cfg, "target_eps_file_name", default="target_eps.pt"))
    )
    target_uncond_eps_path = targets_dir / str(
        OmegaConf.select(
            cfg,
            "target_uncond_eps_file_name",
            default="target_eps_uncond.pt",
        )
    )
    return conditioning_path, target_eps_path, target_uncond_eps_path


def load_trajectory(sample_dir: Path, cfg: DictConfig) -> torch.Tensor:
    latents_dir = sample_dir / str(OmegaConf.select(cfg, "latents_dir_name", default="latents"))
    trajectory_path = latents_dir / str(
        OmegaConf.select(cfg, "latents_file_name", default="trajectory.pt")
    )
    if trajectory_path.exists():
        trajectory = torch.load(trajectory_path, map_location="cpu")
        return squeeze_saved_batch_dim(trajectory)

    latent_paths = sorted(latents_dir.glob("x_*.pt"))
    if not latent_paths:
        raise FileNotFoundError(f"No latent trajectory found in {latents_dir}")
    return torch.stack(
        [squeeze_saved_batch_dim(torch.load(path, map_location="cpu")) for path in latent_paths],
        dim=0,
    )


def squeeze_saved_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def transition_timesteps(sample_dir: Path, trajectory_length: int) -> list[int]:
    timesteps = load_json(sample_dir / "timesteps.json")
    if len(timesteps) == trajectory_length:
        return [int(timestep) for timestep in timesteps[1:]]
    if len(timesteps) == trajectory_length - 1:
        return [int(timestep) for timestep in timesteps]
    raise ValueError(
        f"Unexpected timestep count for {sample_dir}: got {len(timesteps)}, "
        f"expected {trajectory_length} or {trajectory_length - 1}."
    )


@torch.no_grad()
def compute_target_eps(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    trajectory: torch.Tensor,
    timesteps: list[int],
    conditioning: dict[str, torch.Tensor],
    batch_size: int,
    return_uncond: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_noisy = trajectory[:-1]
    if len(timesteps) != x_noisy.shape[0]:
        raise ValueError(
            f"Target timestep count {len(timesteps)} does not match "
            f"transition count {x_noisy.shape[0]}."
        )

    device = pipe.device
    has_sdxl = has_sdxl_conditioning(conditioning)
    prompt_embeds = conditioning["prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
    negative_prompt_embeds = (
        conditioning["negative_prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
        if return_uncond
        else None
    )
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    add_time_ids = None
    if has_sdxl:
        pooled_prompt_embeds = conditioning["pooled_prompt_embeds"].to(
            device=device,
            dtype=pipe.unet.dtype,
        )
        add_time_ids = conditioning["add_time_ids"].to(device=device, dtype=pipe.unet.dtype)
        if return_uncond:
            negative_pooled_prompt_embeds = conditioning["negative_pooled_prompt_embeds"].to(
                device=device,
                dtype=pipe.unet.dtype,
            )

    target_chunks = []
    target_uncond_chunks = []
    for start in range(0, x_noisy.shape[0], batch_size):
        end = min(start + batch_size, x_noisy.shape[0])
        chunk_size = end - start
        latents = x_noisy[start:end].to(device=device, dtype=pipe.unet.dtype)
        timestep = torch.tensor(timesteps[start:end], device=device, dtype=torch.long)
        unet_kwargs = {}
        if return_uncond:
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds are required for CFG branch targets.")
            model_input = torch.cat([latents, latents], dim=0)
            timestep_input = timestep.repeat(2)
            encoder_hidden_states = torch.cat(
                [
                    negative_prompt_embeds.repeat(chunk_size, 1, 1),
                    prompt_embeds.repeat(chunk_size, 1, 1),
                ],
                dim=0,
            )
            if has_sdxl:
                if (
                    pooled_prompt_embeds is None
                    or negative_pooled_prompt_embeds is None
                    or add_time_ids is None
                ):
                    raise ValueError("SDXL CFG branch targets require pooled embeds and time ids.")
                unet_kwargs["added_cond_kwargs"] = {
                    "text_embeds": torch.cat(
                        [
                            negative_pooled_prompt_embeds.repeat(chunk_size, 1),
                            pooled_prompt_embeds.repeat(chunk_size, 1),
                        ],
                        dim=0,
                    ),
                    "time_ids": add_time_ids.repeat(chunk_size * 2, 1),
                }
        else:
            model_input = latents
            timestep_input = timestep
            encoder_hidden_states = prompt_embeds.repeat(chunk_size, 1, 1)
            if has_sdxl:
                if pooled_prompt_embeds is None or add_time_ids is None:
                    raise ValueError("SDXL target eps require pooled embeds and time ids.")
                unet_kwargs["added_cond_kwargs"] = {
                    "text_embeds": pooled_prompt_embeds.repeat(chunk_size, 1),
                    "time_ids": add_time_ids.repeat(chunk_size, 1),
                }

        model_input = pipe.scheduler.scale_model_input(model_input, timestep_input)

        target_eps = pipe.unet(
            model_input,
            timestep_input,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            **unet_kwargs,
        )[0]
        if return_uncond:
            target_uncond_eps, target_cond_eps = target_eps.chunk(2)
            target_uncond_chunks.append(target_uncond_eps.detach().cpu())
            target_chunks.append(target_cond_eps.detach().cpu())
        else:
            target_chunks.append(target_eps.detach().cpu())

    target_cond = torch.cat(target_chunks, dim=0)
    target_uncond = torch.cat(target_uncond_chunks, dim=0) if return_uncond else None
    return target_cond, target_uncond


def update_meta(
    sample_dir: Path,
    conditioning_path: Path,
    target_eps_path: Path,
    target_eps_length: int,
    target_uncond_eps_path: Path | None = None,
    target_uncond_eps_length: int | None = None,
) -> None:
    meta_path = sample_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    meta["training_cache"] = {
        "conditioning_file": conditioning_path.name,
        "targets_dir": target_eps_path.parent.name,
        "target_eps_file": target_eps_path.name,
        "target_eps_length": target_eps_length,
        "target_eps_branch": "conditional",
        "cfg_branch_targets_saved": target_uncond_eps_path is not None,
    }
    if target_uncond_eps_path is not None:
        meta["training_cache"]["target_uncond_eps_file"] = target_uncond_eps_path.name
        meta["training_cache"]["target_uncond_eps_length"] = target_uncond_eps_length
    save_json(meta_path, meta)


def sample_dirs(root_dir: Path, cfg: DictConfig) -> list[Path]:
    dirs = sorted(root_dir.glob(str(OmegaConf.select(cfg, "sample_glob", default="sample_*"))))
    start_index = int(OmegaConf.select(cfg, "start_index", default=0))
    num_samples = OmegaConf.select(cfg, "num_samples", default=None)
    if num_samples is None:
        return dirs[start_index:]
    return dirs[start_index : start_index + int(num_samples)]


def apply_job_spec(cfg: DictConfig) -> None:
    if "job_specs" not in cfg or "job_id" not in cfg:
        return

    job_id = int(cfg.job_id)
    job_spec = cfg.job_specs[job_id]
    cfg.root_dirs = [str(job_spec.root_dir)]
    cfg.start_index = int(job_spec.start_index)
    cfg.num_samples = int(job_spec.num_samples)


def process_sample(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    sample_dir: Path,
    cfg: DictConfig,
) -> str:
    conditioning_path, target_eps_path, target_uncond_eps_path = cache_paths(sample_dir, cfg)
    save_cfg_branch_targets_value = OmegaConf.select(
        cfg,
        "save_cfg_branch_targets",
        default=False,
    )
    if not isinstance(save_cfg_branch_targets_value, bool):
        raise ValueError("save_cfg_branch_targets must be a boolean: true or false.")
    meta_path = sample_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    guidance_scale = float(
        meta.get("guidance_scale", OmegaConf.select(cfg, "model.guidance_scale", default=1.0))
    )
    if guidance_scale > 1.0 and not save_cfg_branch_targets_value:
        raise ValueError(
            f"{sample_dir} was generated with guidance_scale={guidance_scale}; "
            "set save_cfg_branch_targets=true to cache unconditional target eps."
        )
    save_cfg_branch_targets = save_cfg_branch_targets_value

    overwrite = bool(OmegaConf.select(cfg, "overwrite", default=False))
    required_cache_paths = [conditioning_path, target_eps_path]
    if save_cfg_branch_targets:
        required_cache_paths.append(target_uncond_eps_path)
    if all(path.exists() for path in required_cache_paths) and not overwrite:
        return "skipped"

    prompt = load_json(sample_dir / "prompt.json").get("prompt", "")
    if not prompt:
        raise ValueError(f"Missing prompt in {sample_dir / 'prompt.json'}")

    trajectory = load_trajectory(sample_dir, cfg)
    timesteps = transition_timesteps(sample_dir, trajectory.shape[0])
    conditioning = encode_prompt_sdxl(
        pipe,
        prompt=prompt,
        negative_prompt=str(OmegaConf.select(cfg, "negative_prompt", default="")),
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )
    target_eps, target_eps_uncond = compute_target_eps(
        pipe,
        trajectory,
        timesteps,
        conditioning,
        batch_size=int(cfg.batch_size),
        return_uncond=save_cfg_branch_targets,
    )
    (
        conditioning_path,
        target_eps_path,
        target_eps_length,
        target_uncond_eps_path,
        target_uncond_eps_length,
    ) = save_training_cache(
        conditioning,
        target_eps,
        sample_dir,
        cfg,
        target_eps_uncond=target_eps_uncond,
    )
    update_meta(
        sample_dir,
        conditioning_path,
        target_eps_path,
        target_eps_length,
        target_uncond_eps_path,
        target_uncond_eps_length,
    )
    return "written"


@hydra.main(config_path="../../config", config_name="precompute_training_cache", version_base=None)
def main(cfg: DictConfig) -> None:
    apply_job_spec(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    pipe = make_pipe(cfg.model, device)
    pipe.unet.eval()
    if pipe.text_encoder is not None:
        pipe.text_encoder.eval()
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.eval()
    pipe.scheduler.set_timesteps(cfg.model.num_inference_steps, device=device)

    counts = {"written": 0, "skipped": 0}
    for root in cfg.root_dirs:
        root_dir = Path(to_absolute_path(str(root)))
        dirs = sample_dirs(root_dir, cfg)
        logger.info("Precomputing training cache for {} samples in {}", len(dirs), root_dir)
        for sample_dir in tqdm(dirs, desc=f"Caching {root_dir.name}"):
            status = process_sample(pipe, sample_dir, cfg)
            counts[status] = counts.get(status, 0) + 1

    logger.success(
        "Finished training-cache precompute: written={} skipped={}",
        counts.get("written", 0),
        counts.get("skipped", 0),
    )


if __name__ == "__main__":
    main()
