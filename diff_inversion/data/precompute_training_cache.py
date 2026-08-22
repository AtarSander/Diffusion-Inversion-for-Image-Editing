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


def save_conditioning_cache(
    conditioning: dict[str, torch.Tensor],
    conditioning_path: Path,
) -> Path:
    conditioning_path.parent.mkdir(parents=True, exist_ok=True)
    conditioning_to_save = {
        "prompt_embeds": conditioning["prompt_embeds"].detach().cpu(),
        "negative_prompt_embeds": conditioning["negative_prompt_embeds"].detach().cpu(),
    }
    if has_sdxl_conditioning(conditioning):
        conditioning_to_save.update(
            {
                "pooled_prompt_embeds": conditioning["pooled_prompt_embeds"].detach().cpu(),
                "negative_pooled_prompt_embeds": conditioning[
                    "negative_pooled_prompt_embeds"
                ].detach().cpu(),
                "add_time_ids": conditioning["add_time_ids"].detach().cpu(),
            }
        )
    torch.save(conditioning_to_save, conditioning_path)
    return conditioning_path


def save_target_tensor(tensor: torch.Tensor, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu()
    torch.save(tensor, path)
    return int(tensor.shape[0])


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
    target_branches: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    include_cond = target_branches in {"conditional", "both"}
    include_uncond = target_branches in {"unconditional", "both"}
    if not include_cond and not include_uncond:
        raise ValueError(f"Unsupported target_branches={target_branches!r}.")

    x_noisy = trajectory[:-1]
    if len(timesteps) != x_noisy.shape[0]:
        raise ValueError(
            f"Target timestep count {len(timesteps)} does not match "
            f"transition count {x_noisy.shape[0]}."
        )

    device = pipe.device
    has_sdxl = has_sdxl_conditioning(conditioning)
    prompt_embeds = (
        conditioning["prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
        if include_cond
        else None
    )
    negative_prompt_embeds = (
        conditioning["negative_prompt_embeds"].to(device=device, dtype=pipe.unet.dtype)
        if include_uncond
        else None
    )
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    add_time_ids = None
    if has_sdxl:
        pooled_prompt_embeds = (
            conditioning["pooled_prompt_embeds"].to(
                device=device,
                dtype=pipe.unet.dtype,
            )
            if include_cond
            else None
        )
        add_time_ids = conditioning["add_time_ids"].to(device=device, dtype=pipe.unet.dtype)
        if include_uncond:
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
        model_inputs = []
        timestep_inputs = []
        encoder_hidden_states = []
        text_embeds = []
        time_ids = []

        if include_uncond:
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds are required for unconditional targets.")
            model_inputs.append(latents)
            timestep_inputs.append(timestep)
            encoder_hidden_states.append(negative_prompt_embeds.repeat(chunk_size, 1, 1))
            if has_sdxl:
                if negative_pooled_prompt_embeds is None or add_time_ids is None:
                    raise ValueError("SDXL unconditional targets require pooled embeds and time ids.")
                text_embeds.append(negative_pooled_prompt_embeds.repeat(chunk_size, 1))
                time_ids.append(add_time_ids.repeat(chunk_size, 1))

        if include_cond:
            if prompt_embeds is None:
                raise ValueError("prompt_embeds are required for conditional targets.")
            model_inputs.append(latents)
            timestep_inputs.append(timestep)
            encoder_hidden_states.append(prompt_embeds.repeat(chunk_size, 1, 1))
            if has_sdxl:
                if pooled_prompt_embeds is None or add_time_ids is None:
                    raise ValueError("SDXL target eps require pooled embeds and time ids.")
                text_embeds.append(pooled_prompt_embeds.repeat(chunk_size, 1))
                time_ids.append(add_time_ids.repeat(chunk_size, 1))

        model_input = torch.cat(model_inputs, dim=0)
        timestep_input = torch.cat(timestep_inputs, dim=0)
        encoder_hidden_states_input = torch.cat(encoder_hidden_states, dim=0)
        if has_sdxl:
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": torch.cat(text_embeds, dim=0),
                "time_ids": torch.cat(time_ids, dim=0),
            }

        model_input = pipe.scheduler.scale_model_input(model_input, timestep_input)

        target_eps = pipe.unet(
            model_input,
            timestep_input,
            encoder_hidden_states=encoder_hidden_states_input,
            return_dict=False,
            **unet_kwargs,
        )[0]
        cursor = 0
        if include_uncond:
            target_uncond_chunks.append(target_eps[cursor : cursor + chunk_size].detach().cpu())
            cursor += chunk_size
        if include_cond:
            target_chunks.append(target_eps[cursor : cursor + chunk_size].detach().cpu())

    target_cond = torch.cat(target_chunks, dim=0) if include_cond else None
    target_uncond = torch.cat(target_uncond_chunks, dim=0) if include_uncond else None
    return target_cond, target_uncond


def update_meta(
    sample_dir: Path,
    conditioning_path: Path,
    target_eps_path: Path,
    target_eps_length: int | None = None,
    target_uncond_eps_path: Path | None = None,
    target_uncond_eps_length: int | None = None,
) -> None:
    meta_path = sample_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    training_cache = dict(meta.get("training_cache", {}))
    if target_eps_length is None and target_eps_path.exists():
        target_eps_length = int(torch.load(target_eps_path, map_location="cpu").shape[0])

    training_cache.update(
        {
            "conditioning_file": conditioning_path.name,
            "targets_dir": target_eps_path.parent.name,
            "target_eps_file": target_eps_path.name,
            "target_eps_length": target_eps_length,
            "target_eps_branch": "conditional",
        }
    )
    if target_uncond_eps_path is not None:
        training_cache.update(
            {
                "cfg_branch_targets_saved": True,
                "target_uncond_eps_file": target_uncond_eps_path.name,
                "target_uncond_eps_length": target_uncond_eps_length,
            }
        )
    else:
        training_cache.setdefault("cfg_branch_targets_saved", False)

    meta["training_cache"] = training_cache
    save_json(meta_path, meta)


def target_branches_from_config(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "target_branches", default=None)
    if value is None:
        save_cfg_branch_targets = OmegaConf.select(
            cfg,
            "save_cfg_branch_targets",
            default=False,
        )
        if not isinstance(save_cfg_branch_targets, bool):
            raise ValueError("save_cfg_branch_targets must be a boolean: true or false.")
        return "both" if save_cfg_branch_targets else "conditional"

    branches = str(value).strip().lower()
    aliases = {
        "cond": "conditional",
        "text": "conditional",
        "uncond": "unconditional",
        "null": "unconditional",
        "cfg": "both",
    }
    branches = aliases.get(branches, branches)
    if branches not in {"conditional", "unconditional", "both"}:
        raise ValueError(
            "target_branches must be one of: conditional, unconditional, both; "
            f"got {value!r}."
        )
    return branches


def conditioning_has_required_negative_tensors(conditioning: dict[str, torch.Tensor]) -> bool:
    if "negative_prompt_embeds" not in conditioning:
        return False
    if has_sdxl_conditioning(conditioning):
        return "negative_pooled_prompt_embeds" in conditioning
    return True


def load_or_encode_conditioning(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    sample_dir: Path,
    conditioning_path: Path,
    cfg: DictConfig,
    prompt: str,
    target_branches: str,
) -> dict[str, torch.Tensor]:
    if conditioning_path.exists():
        conditioning = torch.load(conditioning_path, map_location="cpu")
        needs_negative_conditioning = target_branches in {"unconditional", "both"}
        if (
            not needs_negative_conditioning
            or conditioning_has_required_negative_tensors(conditioning)
        ):
            return conditioning

    conditioning = encode_prompt_sdxl(
        pipe,
        prompt=prompt,
        negative_prompt=str(OmegaConf.select(cfg, "negative_prompt", default="")),
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )
    save_conditioning_cache(conditioning, conditioning_path)
    logger.info("Wrote conditioning cache for {}", sample_dir)
    return conditioning


def validate_existing_conditional_target(
    target_eps_path: Path,
    target_uncond_eps: torch.Tensor,
    sample_dir: Path,
    cfg: DictConfig,
) -> int:
    if not bool(OmegaConf.select(cfg, "require_existing_conditional_target", default=True)):
        return int(target_uncond_eps.shape[0])
    if not target_eps_path.exists():
        raise FileNotFoundError(
            "target_branches=unconditional requires existing conditional target eps: "
            f"{target_eps_path}"
        )
    target_eps_length = int(torch.load(target_eps_path, map_location="cpu").shape[0])
    if target_eps_length != int(target_uncond_eps.shape[0]):
        raise ValueError(
            f"Conditional/unconditional target length mismatch in {sample_dir}: "
            f"{target_eps_length} vs {int(target_uncond_eps.shape[0])}."
        )
    return target_eps_length


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
    target_branches = target_branches_from_config(cfg)
    include_cond = target_branches in {"conditional", "both"}
    include_uncond = target_branches in {"unconditional", "both"}

    meta_path = sample_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    guidance_scale = float(
        meta.get("guidance_scale", OmegaConf.select(cfg, "model.guidance_scale", default=1.0))
    )
    if guidance_scale > 1.0 and target_branches == "conditional":
        raise ValueError(
            f"{sample_dir} was generated with guidance_scale={guidance_scale}; "
            "set target_branches=unconditional or target_branches=both to cache "
            "the CFG branch target eps."
        )

    overwrite = bool(OmegaConf.select(cfg, "overwrite", default=False))
    required_cache_paths: list[Path] = []
    if include_cond:
        required_cache_paths.extend([conditioning_path, target_eps_path])
    if include_uncond:
        required_cache_paths.extend([conditioning_path, target_uncond_eps_path])
        if target_branches == "unconditional" and bool(
            OmegaConf.select(cfg, "require_existing_conditional_target", default=True)
        ):
            required_cache_paths.append(target_eps_path)
    if all(path.exists() for path in required_cache_paths) and not overwrite:
        return "skipped"

    if target_branches == "unconditional" and bool(
        OmegaConf.select(cfg, "require_existing_conditional_target", default=True)
    ):
        if not target_eps_path.exists():
            raise FileNotFoundError(
                "target_branches=unconditional requires existing conditional target eps: "
                f"{target_eps_path}"
            )

    prompt = load_json(sample_dir / "prompt.json").get("prompt", "")
    if not prompt:
        raise ValueError(f"Missing prompt in {sample_dir / 'prompt.json'}")

    trajectory = load_trajectory(sample_dir, cfg)
    timesteps = transition_timesteps(sample_dir, trajectory.shape[0])
    conditioning = load_or_encode_conditioning(
        pipe,
        sample_dir,
        conditioning_path,
        cfg,
        prompt=prompt,
        target_branches=target_branches,
    )
    target_eps, target_eps_uncond = compute_target_eps(
        pipe,
        trajectory,
        timesteps,
        conditioning,
        batch_size=int(cfg.batch_size),
        target_branches=target_branches,
    )

    if target_branches == "unconditional":
        if target_eps is not None or target_eps_uncond is None:
            raise RuntimeError("Expected only unconditional target eps.")
        target_uncond_eps_length = save_target_tensor(target_eps_uncond, target_uncond_eps_path)
        target_eps_length = validate_existing_conditional_target(
            target_eps_path,
            target_eps_uncond,
            sample_dir,
            cfg,
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

    if not include_cond or target_eps is None:
        raise RuntimeError(f"Expected conditional target eps for target_branches={target_branches}.")
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

    target_branches = target_branches_from_config(cfg)
    logger.info(
        "Precompute settings: job_id={} target_branches={} targets_dir_name={} "
        "overwrite={} require_existing_conditional_target={} batch_size={} "
        "start_index={} num_samples={} root_dirs={}",
        OmegaConf.select(cfg, "job_id", default=None),
        target_branches,
        OmegaConf.select(cfg, "targets_dir_name", default="targets"),
        OmegaConf.select(cfg, "overwrite", default=False),
        OmegaConf.select(cfg, "require_existing_conditional_target", default=True),
        OmegaConf.select(cfg, "batch_size", default=None),
        OmegaConf.select(cfg, "start_index", default=0),
        OmegaConf.select(cfg, "num_samples", default=None),
        list(cfg.root_dirs),
    )

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
    total_processed = 0
    for root in cfg.root_dirs:
        root_dir = Path(to_absolute_path(str(root)))
        dirs = sample_dirs(root_dir, cfg)
        logger.info(
            "Precomputing training cache for {} samples in {}",
            len(dirs),
            root_dir,
        )
        root_counts: dict[str, int] = {}
        root_processed = 0
        for sample_dir in tqdm(dirs, desc=f"Caching {root_dir.name}"):
            status = process_sample(pipe, sample_dir, cfg)
            counts[status] = counts.get(status, 0) + 1
            root_counts[status] = root_counts.get(status, 0) + 1
            root_processed += 1
            total_processed += 1
        logger.info(
            "Finished root {}: processed={} statuses={}",
            root_dir,
            root_processed,
            root_counts,
        )

    logger.success(
        "Finished training-cache precompute: processed={} statuses={}",
        total_processed,
        counts,
    )
    logger.success(
        "Finished training-cache precompute: written={} skipped={}",
        counts.get("written", 0),
        counts.get("skipped", 0),
    )


if __name__ == "__main__":
    main()
