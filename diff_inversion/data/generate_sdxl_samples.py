"""Generate SDXL samples and latent trajectories from prepared prompt files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from diffusers import StableDiffusionXLPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

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


@torch.no_grad()
def decode_latent_to_pil(pipe: StableDiffusionXLPipeline, latents: torch.Tensor) -> Image.Image:
    """Decode a latent tensor into a PIL image."""
    pipe.vae.to(dtype=torch.float32)
    latents_fp32 = latents.to(device=pipe.device, dtype=torch.float32)

    decoded = pipe.vae.decode(
        latents_fp32 / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]

    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image


@torch.no_grad()
def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
) -> Dict[str, torch.Tensor]:
    """Encode prompt text and auxiliary conditioning tensors for SDXL."""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=pipe._execution_device,
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


@torch.no_grad()
def sample_with_trajectory(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[int]]:
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
    timestep_values: List[int] = [
        int(timesteps[0].item()) if hasattr(timesteps[0], "item") else int(timesteps[0])
    ]

    encoder_hidden_states = torch.cat(
        [cond["negative_prompt_embeds"], cond["prompt_embeds"]],
        dim=0,
    )
    text_embeds = torch.cat(
        [cond["negative_pooled_prompt_embeds"], cond["pooled_prompt_embeds"]],
        dim=0,
    )
    time_ids = cond["add_time_ids"].repeat(2, 1)

    for t in tqdm(timesteps, desc="Denoising", leave=False):
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
            return_dict=False,
        )[0]

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            return_dict=True,
        ).prev_sample

        trajectory.append(latents.detach().cpu())
        pred_noises.append(noise_pred.detach().cpu())
        timestep_values.append(int(t.item()) if hasattr(t, "item") else int(t))

    return latents, trajectory, pred_noises, timestep_values


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

    if gather_cfg.save_noises:
        pred_noises_dir.mkdir(parents=True, exist_ok=True)

    prompt = record["prompt"]
    seed = gather_cfg.seed + sample_idx

    final_latent, trajectory, pred_noises, timestep_values = sample_with_trajectory(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=gather_cfg.negative_prompt,
        num_inference_steps=model_cfg.num_inference_steps,
        guidance_scale=model_cfg.guidance_scale,
        height=model_cfg.height,
        width=model_cfg.width,
        seed=seed,
    )

    if gather_cfg.save_final_image:
        final_image = decode_latent_to_pil(pipe, final_latent)
        final_image.save(sample_dir / str(gather_cfg.final_image_name))

    if gather_cfg.save_latents:
        for i, latent in enumerate(trajectory):
            torch.save(latent, latents_dir / f"x_{i:03d}.pt")

    if gather_cfg.save_pred_noises:
        for i, noise in enumerate(pred_noises):
            torch.save(noise, pred_noises_dir / f"noise_{i:03d}.pt")

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
    if gather_cfg.save_meta:
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if gather_cfg.save_timesteps:
        with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
            json.dump(timestep_values, f, indent=2)

    logger.success("Saved sample {}: {}", sample_idx, sample_dir)


@hydra.main(config_path="../../config", config_name="sample_gather", version_base=None)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for generating SDXL samples from prepared prompts."""
    model_cfg = cfg.model
    gather_cfg = cfg

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
    logger.info(
        "Generating {} samples from records [{}:{}) into {}",
        len(prompts),
        start_index,
        end_index,
        out_dir,
    )

    pipe = make_pipe(model_cfg, device)

    with (out_dir / str(gather_cfg.run_config_name)).open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)
    logger.info("Saved run config: {}", out_dir / str(gather_cfg.run_config_name))

    for sample_idx, record in tqdm(
        enumerate(prompts, start=start_index),
        total=len(prompts),
        desc="Generating samples",
    ):
        save_sample(pipe, record, sample_idx, model_cfg, gather_cfg, out_dir)

    logger.success("Finished generating {} samples", len(prompts))


if __name__ == "__main__":
    main()
