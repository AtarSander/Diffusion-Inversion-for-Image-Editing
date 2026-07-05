"""Generate SDXL samples and latent trajectories from prepared prompt files."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl_branches
from diff_inversion.utils import make_pipe


def load_recap_prompt_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load prepared Recap-COCO prompt records from a JSON Lines file."""
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def nested_get(data: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_existing_run_config(out_dir: Path, cfg: DictConfig) -> None:
    """Refuse to resume into a directory generated with incompatible settings."""
    if bool(OmegaConf.select(cfg, "overwrite", default=False)):
        return

    run_config_path = out_dir / str(
        OmegaConf.select(cfg, "run_config_name", default="run_config.json")
    )
    if not run_config_path.exists():
        return

    with run_config_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    if not isinstance(existing, dict):
        raise ValueError(f"Expected dict in existing run config: {run_config_path}")

    checked_keys = [
        "data.prompts_jsonl",
        "model.model_id",
        "model.scheduler",
        "model.num_inference_steps",
        "model.guidance_scale",
        "model.height",
        "model.width",
        "negative_prompt",
        "seed",
        "save_training_cache",
        "save_cfg_branch_targets",
        "latents_file_name",
        "targets_dir_name",
        "target_eps_file_name",
        "target_uncond_eps_file_name",
    ]
    mismatches = []
    for key in checked_keys:
        current_value = OmegaConf.select(cfg, key, default=None)
        existing_value = nested_get(existing, key)
        if str(existing_value) != str(current_value):
            mismatches.append(
                f"{key}: existing={existing_value!r}, current={current_value!r}"
            )

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            f"Output directory already has an incompatible {run_config_path.name}: {out_dir}\n"
            f"  - {joined}\n"
            "Use a new OUTPUT_DIR for this CFG, or set overwrite=true intentionally."
        )


def save_run_config(path: Path, cfg: DictConfig) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


@torch.no_grad()
def decode_latent_to_pil(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline, latents: torch.Tensor
) -> Image.Image:
    """Decode a latent tensor into a PIL image."""
    pipe.vae.to(dtype=torch.float32)
    latents_fp32 = latents.to(device=pipe.device, dtype=torch.float32)

    decoded = pipe.vae.decode(
        latents_fp32 / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]

    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image


def is_sdxl_pipeline(pipe: StableDiffusionPipeline | StableDiffusionXLPipeline) -> bool:
    return getattr(pipe, "text_encoder_2", None) is not None


def has_sdxl_conditioning(conditioning: Dict[str, torch.Tensor]) -> bool:
    return "pooled_prompt_embeds" in conditioning and "add_time_ids" in conditioning


def _pipeline_device(pipe: StableDiffusionPipeline | StableDiffusionXLPipeline) -> torch.device:
    return getattr(pipe, "_execution_device", pipe.device)


@torch.no_grad()
def encode_prompt_sdxl(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
) -> Dict[str, torch.Tensor]:
    """Encode prompt text and optional SDXL auxiliary conditioning tensors."""
    if is_sdxl_pipeline(pipe):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=_pipeline_device(pipe),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )

        add_time_ids = pipe._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(pipe.device)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "add_time_ids": add_time_ids,
        }

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=_pipeline_device(pipe),
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }


@torch.no_grad()
def sample_with_trajectory(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    save_cfg_branch_targets: bool,
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor] | None,
    Dict[str, torch.Tensor],
    List[int],
]:
    """Run DDIM sampling and keep the full latent trajectory."""
    device = pipe.device

    # Keep batch size at 1 so each prompt produces an independent latent trajectory
    # with its own sample directory and per-step tensors.
    batch_size = 1

    cond = encode_prompt_sdxl(pipe, prompt, negative_prompt, height, width)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=pipe.unet.dtype,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    trajectory: List[torch.Tensor] = [latents.detach().cpu()]
    pred_noises: List[torch.Tensor] = []
    target_eps: List[torch.Tensor] = []
    target_eps_uncond: List[torch.Tensor] | None = [] if save_cfg_branch_targets else None
    timestep_values: List[int] = [
        int(timesteps[0].item()) if hasattr(timesteps[0], "item") else int(timesteps[0])
    ]

    for t in tqdm(timesteps, desc="Denoising", leave=False):
        noise_uncond, noise_text, noise_pred = predict_noise_sdxl_branches(
            pipe=pipe,
            latents=latents,
            timestep=t,
            cond=cond,
            guidance_scale=guidance_scale,
        )
        target_eps.append(noise_text.detach().cpu())
        if target_eps_uncond is not None:
            target_eps_uncond.append(noise_uncond.detach().cpu())

        latents = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            return_dict=True,
        ).prev_sample

        trajectory.append(latents.detach().cpu())
        pred_noises.append(noise_pred.detach().cpu())
        timestep_values.append(int(t.item()) if hasattr(t, "item") else int(t))

    return latents, trajectory, pred_noises, target_eps, target_eps_uncond, cond, timestep_values


def save_training_cache(
    conditioning: Dict[str, torch.Tensor],
    target_eps: List[torch.Tensor] | torch.Tensor,
    sample_dir: Path,
    cfg: DictConfig,
    target_eps_uncond: List[torch.Tensor] | torch.Tensor | None = None,
) -> tuple[Path, Path, int, Path | None, int | None]:
    """Persist cached conditioning and target noise for LoRA inversion training."""
    conditioning_path = sample_dir / str(
        OmegaConf.select(cfg, "conditioning_file_name", default="conditioning.pt")
    )
    targets_dir = sample_dir / str(OmegaConf.select(cfg, "targets_dir_name", default="targets"))
    target_eps_path = targets_dir / str(
        OmegaConf.select(cfg, "target_eps_file_name", default="target_eps.pt")
    )
    targets_dir.mkdir(parents=True, exist_ok=True)

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

    if isinstance(target_eps, torch.Tensor):
        target_eps_tensor = target_eps.detach().cpu()
    else:
        target_eps_tensor = torch.cat([eps.detach().cpu() for eps in target_eps], dim=0)
    torch.save(target_eps_tensor, target_eps_path)

    target_uncond_eps_path = None
    target_uncond_eps_length = None
    if target_eps_uncond is not None:
        target_uncond_eps_path = targets_dir / str(
            OmegaConf.select(
                cfg,
                "target_uncond_eps_file_name",
                default="target_eps_uncond.pt",
            )
        )
        if isinstance(target_eps_uncond, torch.Tensor):
            target_uncond_eps_tensor = target_eps_uncond.detach().cpu()
        else:
            target_uncond_eps_tensor = torch.cat(
                [eps.detach().cpu() for eps in target_eps_uncond],
                dim=0,
            )
        if target_uncond_eps_tensor.shape[0] != target_eps_tensor.shape[0]:
            raise ValueError(
                "Conditional and unconditional target eps lengths do not match: "
                f"{target_eps_tensor.shape[0]} vs {target_uncond_eps_tensor.shape[0]}."
            )
        torch.save(target_uncond_eps_tensor, target_uncond_eps_path)
        target_uncond_eps_length = int(target_uncond_eps_tensor.shape[0])

    return (
        conditioning_path,
        target_eps_path,
        int(target_eps_tensor.shape[0]),
        target_uncond_eps_path,
        target_uncond_eps_length,
    )


def save_latent_trajectory(
    trajectory: List[torch.Tensor],
    latents_dir: Path,
    gather_cfg: DictConfig,
) -> str:
    """Persist latent trajectory tensors using the configured file layout."""
    latents_format = str(OmegaConf.select(gather_cfg, "latents_format", default="stacked_pt"))

    if latents_format == "per_step_pt":
        template = str(
            OmegaConf.select(
                gather_cfg,
                "per_step_latent_template",
                default="x_{step_idx:03d}.pt",
            )
        )
        for step_idx, latent in enumerate(trajectory):
            torch.save(latent, latents_dir / template.format(step_idx=step_idx))
        return latents_format

    if latents_format == "stacked_pt":
        file_name = str(OmegaConf.select(gather_cfg, "latents_file_name", default="trajectory.pt"))
        torch.save(torch.stack(trajectory, dim=0), latents_dir / file_name)
        return latents_format

    raise ValueError(
        f"Unsupported latents_format: {latents_format}. Expected one of: stacked_pt, per_step_pt."
    )


def save_sample(
    pipe: StableDiffusionXLPipeline,
    record: Dict[str, Any],
    sample_idx: int,
    model_cfg: DictConfig,
    gather_cfg: DictConfig,
    out_dir: Path,
) -> None:
    """Generate and persist one sample directory with images, latents, and metadata."""
    sample_dir = out_dir / gather_cfg.sample_dir_template.format(sample_idx=sample_idx)
    latents_dir = sample_dir / str(gather_cfg.latents_dir_name)
    pred_noises_dir = sample_dir / str(gather_cfg.pred_noises_dir_name)

    if sample_dir.exists() and not gather_cfg.overwrite:
        logger.info("Skipping existing sample: {}", sample_dir)
        return

    sample_dir.mkdir(parents=True, exist_ok=True)
    if gather_cfg.save_latents:
        latents_dir.mkdir(parents=True, exist_ok=True)

    save_pred_noises = bool(OmegaConf.select(gather_cfg, "save_pred_noises", default=False))
    save_training_cache_enabled = bool(
        OmegaConf.select(gather_cfg, "save_training_cache", default=True)
    )

    if save_pred_noises:
        pred_noises_dir.mkdir(parents=True, exist_ok=True)

    prompt = record["prompt"]
    seed = gather_cfg.seed + sample_idx

    save_cfg_branch_targets_value = OmegaConf.select(
        gather_cfg,
        "save_cfg_branch_targets",
        default=False,
    )
    if not isinstance(save_cfg_branch_targets_value, bool):
        raise ValueError("save_cfg_branch_targets must be a boolean: true or false.")
    if (
        save_training_cache_enabled
        and float(model_cfg.guidance_scale) > 1.0
        and not save_cfg_branch_targets_value
    ):
        raise ValueError(
            "model.guidance_scale > 1.0 requires save_cfg_branch_targets=true "
            "so unconditional target eps are saved."
        )
    save_cfg_branch_targets = save_training_cache_enabled and save_cfg_branch_targets_value
    (
        final_latent,
        trajectory,
        pred_noises,
        target_eps,
        target_eps_uncond,
        conditioning,
        timestep_values,
    ) = sample_with_trajectory(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=gather_cfg.negative_prompt,
        num_inference_steps=model_cfg.num_inference_steps,
        guidance_scale=model_cfg.guidance_scale,
        height=model_cfg.height,
        width=model_cfg.width,
        seed=seed,
        save_cfg_branch_targets=save_cfg_branch_targets,
    )

    if gather_cfg.save_final_image:
        final_image = decode_latent_to_pil(pipe, final_latent)
        final_image.save(sample_dir / str(gather_cfg.final_image_name))

    latents_format = None
    if gather_cfg.save_latents:
        latents_format = save_latent_trajectory(trajectory, latents_dir, gather_cfg)

    if save_pred_noises:
        for i, noise in enumerate(pred_noises):
            torch.save(noise, pred_noises_dir / f"noise_{i:03d}.pt")

    training_cache_meta = None
    if save_training_cache_enabled:
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
            gather_cfg,
            target_eps_uncond=target_eps_uncond,
        )
        training_cache_meta = {
            "conditioning_file": conditioning_path.name,
            "targets_dir": target_eps_path.parent.name,
            "target_eps_file": target_eps_path.name,
            "target_eps_length": target_eps_length,
            "target_eps_branch": "conditional",
            "cfg_branch_targets_saved": target_uncond_eps_path is not None,
        }
        if target_uncond_eps_path is not None:
            training_cache_meta["target_uncond_eps_file"] = target_uncond_eps_path.name
            training_cache_meta["target_uncond_eps_length"] = target_uncond_eps_length

    if gather_cfg.save_prompt:
        with (sample_dir / "prompt.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    meta = {
        "sample_idx": sample_idx,
        "seed": seed,
        "model_id": model_cfg.model_id,
        "num_inference_steps": model_cfg.num_inference_steps,
        "guidance_scale": model_cfg.guidance_scale,
        "height": model_cfg.height,
        "width": model_cfg.width,
        "negative_prompt": gather_cfg.negative_prompt,
        "trajectory_length": len(trajectory),
        "pred_noises_length": len(pred_noises),
    }
    if latents_format is not None:
        meta["latents_format"] = latents_format
    if training_cache_meta is not None:
        meta["training_cache"] = training_cache_meta
    if gather_cfg.save_meta:
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if gather_cfg.save_timesteps:
        with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
            json.dump(timestep_values, f, indent=2)

    logger.success("Saved sample {}: {}", sample_idx, sample_dir)


def apply_job_spec(gather_cfg: DictConfig) -> None:
    """Apply optional per-job overrides from sample_gather_*_submitit configs."""
    if "job_specs" not in gather_cfg or "job_id" not in gather_cfg:
        return

    job_id = int(gather_cfg.job_id)
    job_spec = gather_cfg.job_specs[job_id]
    gather_cfg.start_index = int(job_spec.start_index)
    gather_cfg.num_samples = int(job_spec.num_samples)

    if "output_dir" in job_spec:
        gather_cfg.output_dir = str(job_spec.output_dir)
    if "prompts_jsonl" in job_spec:
        gather_cfg.data.prompts_jsonl = str(job_spec.prompts_jsonl)


@hydra.main(config_path="../../config", config_name="sample_gather_submitit", version_base=None)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for generating SDXL samples from prepared prompts."""
    model_cfg = cfg.model
    gather_cfg = cfg

    apply_job_spec(gather_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_cfg.require_cuda and device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    prompts_jsonl = Path(to_absolute_path(str(cfg.data.prompts_jsonl)))
    if not prompts_jsonl.exists():
        raise FileNotFoundError(
            f"Prompt JSONL not found: {prompts_jsonl}\n"
            "Run `make data-prepare-recap-coco` first or override data.prompts_jsonl."
        )

    logger.info("Loading Recap-COCO prompt records from {}", prompts_jsonl)
    prompts = load_recap_prompt_records(prompts_jsonl)
    if not prompts:
        raise ValueError("No prompts found in the configured dataset.")

    start_index = int(gather_cfg.start_index)
    end_index = start_index + int(gather_cfg.num_samples)
    prompts = prompts[start_index:end_index]
    out_dir = Path(to_absolute_path(str(gather_cfg.output_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_existing_run_config(out_dir, cfg)
    logger.info(
        "Generating {} samples from records [{}:{}) into {} with guidance_scale={} "
        "save_training_cache={} save_cfg_branch_targets={}",
        len(prompts),
        start_index,
        end_index,
        out_dir,
        model_cfg.guidance_scale,
        OmegaConf.select(gather_cfg, "save_training_cache", default=True),
        OmegaConf.select(gather_cfg, "save_cfg_branch_targets", default=False),
    )

    pipe = make_pipe(model_cfg, device)

    run_config_path = out_dir / str(gather_cfg.run_config_name)
    save_run_config(run_config_path, cfg)
    logger.info("Saved run config: {}", run_config_path)

    for sample_idx, record in tqdm(
        enumerate(prompts, start=start_index),
        total=len(prompts),
        desc="Generating samples",
    ):
        save_sample(pipe, record, sample_idx, model_cfg, gather_cfg, out_dir)

    logger.success("Finished generating {} samples", len(prompts))


if __name__ == "__main__":
    main()
