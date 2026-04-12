"""Generate SDXL samples and latent trajectories from prepared prompt files."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image
import torch


@dataclass
class SampleConfig:
    """Configuration for generating SDXL samples from a prompt dataset."""

    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "sdxl_samples"
    prompts_jsonl: str = "train.jsonl"
    num_samples: int = 4
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    seed: int = 1234
    save_latents: bool = True
    overwrite: bool = False


def load_prompts(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load prompt records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_pipe(cfg: SampleConfig, device: str) -> StableDiffusionXLPipeline:
    """Create an SDXL pipeline configured for DDIM sampling."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe


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
) -> dict[str, torch.Tensor]:
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
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run DDIM sampling and keep the full latent trajectory."""
    device = pipe.device
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

    trajectory: list[torch.Tensor] = [latents.detach().cpu()]
    timestep_values: list[int] = [
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

    for t in timesteps:
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
        timestep_values.append(int(t.item()) if hasattr(t, "item") else int(t))

    return latents, trajectory, timestep_values


def save_sample(
    pipe: StableDiffusionXLPipeline,
    record: dict[str, Any],
    sample_idx: int,
    cfg: SampleConfig,
    out_dir: Path,
) -> None:
    """Generate and persist one sample directory with images, latents, and metadata."""
    sample_dir = out_dir / f"sample_{sample_idx:06d}"
    latents_dir = sample_dir / "latents"

    if sample_dir.exists() and not cfg.overwrite:
        print(f"Skipping existing sample: {sample_dir}")
        return

    sample_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    prompt = record["prompt"]
    seed = cfg.seed + sample_idx

    final_latent, trajectory, timestep_values = sample_with_trajectory(
        pipe=pipe,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        height=cfg.height,
        width=cfg.width,
        seed=seed,
    )

    final_image = decode_latent_to_pil(pipe, final_latent)
    final_image.save(sample_dir / "final.png")

    if cfg.save_latents:
        for i, latent in enumerate(trajectory):
            torch.save(latent, latents_dir / f"x_{i:03d}.pt")

    with (sample_dir / "prompt.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    meta = {
        "sample_idx": sample_idx,
        "seed": seed,
        "model_id": cfg.model_id,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "height": cfg.height,
        "width": cfg.width,
        "trajectory_length": len(trajectory),
    }
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
        json.dump(timestep_values, f, indent=2)

    print(f"Saved sample {sample_idx}: {sample_dir}")


def main() -> None:
    """CLI entrypoint for generating SDXL samples from prepared prompts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sdxl_samples")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = SampleConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        prompts_jsonl=args.prompts_jsonl,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This script is intended to run on CUDA.")

    prompts = load_prompts(Path(cfg.prompts_jsonl))
    if not prompts:
        raise ValueError("No prompts found in the provided JSONL file.")

    prompts = prompts[: cfg.num_samples]
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = make_pipe(cfg, device)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    for sample_idx, record in enumerate(prompts):
        save_sample(pipe, record, sample_idx, cfg, out_dir)


if __name__ == "__main__":
    main()
