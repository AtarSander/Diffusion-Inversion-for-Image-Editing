#!/usr/bin/env python
"""Smoke-check SD1.5 LoRA inversion target-cache alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from diff_inversion.utils import make_pipe


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def squeeze_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.ndim == 5 and trajectory.shape[1] == 1:
        return trajectory[:, 0]
    if trajectory.ndim == 4:
        return trajectory
    raise ValueError(f"Expected trajectory [T,1,C,H,W] or [T,C,H,W], got {tuple(trajectory.shape)}")


def transition_timestep(timesteps: list[int], step_idx: int, trajectory_length: int) -> int:
    if len(timesteps) == trajectory_length:
        return int(timesteps[step_idx + 1])
    if len(timesteps) == trajectory_length - 1:
        return int(timesteps[step_idx])
    raise ValueError(
        f"Unexpected timestep count: got {len(timesteps)}, "
        f"expected {trajectory_length} or {trajectory_length - 1}"
    )


@torch.no_grad()
def predict_eps(pipe, latents: torch.Tensor, timestep: int, prompt_embeds: torch.Tensor) -> torch.Tensor:
    latents = latents.to(device=pipe.device, dtype=pipe.unet.dtype)
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    timesteps = torch.full((latents.shape[0],), int(timestep), device=pipe.device, dtype=torch.long)
    prompt_embeds = prompt_embeds.to(device=pipe.device, dtype=pipe.unet.dtype)
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.shape[0] == 1 and latents.shape[0] > 1:
        prompt_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)

    model_input = pipe.scheduler.scale_model_input(latents, timesteps)
    return pipe.unet(
        model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0].detach().float().cpu()


def sample_dirs(root: Path, max_samples: int) -> list[Path]:
    dirs = sorted(path for path in root.glob("sample_*") if path.is_dir())
    return dirs[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("data/processed/sd15_trajectories_stacked_fp32/all"),
    )
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg = OmegaConf.load("config/model/sd15.yaml")

    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run this smoke test on a GPU node.")

    pipe = make_pipe(model_cfg, args.device)
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(int(model_cfg.num_inference_steps), device=args.device)

    rows = []
    for sample_dir in sample_dirs(args.root_dir, args.max_samples):
        trajectory = squeeze_trajectory(load_tensor(sample_dir / "latents" / "trajectory.pt"))
        target_eps = load_tensor(sample_dir / "targets" / "target_eps.pt").float()
        conditioning = load_tensor(sample_dir / "conditioning.pt")
        prompt_embeds = conditioning["prompt_embeds"]
        with (sample_dir / "timesteps.json").open("r", encoding="utf-8") as f:
            timesteps = json.load(f)

        num_steps = min(args.max_steps, int(target_eps.shape[0]), int(trajectory.shape[0]) - 1)
        for step_idx in range(num_steps):
            timestep = transition_timestep(timesteps, step_idx, int(trajectory.shape[0]))
            target = target_eps[step_idx].unsqueeze(0)

            eps_current = predict_eps(pipe, trajectory[step_idx], timestep, prompt_embeds)
            eps_next = predict_eps(pipe, trajectory[step_idx + 1], timestep, prompt_embeds)

            row = {
                "sample": sample_dir.name,
                "step_idx": step_idx,
                "timestep": timestep,
                "mse_current_latent_vs_cache": float(F.mse_loss(eps_current, target).item()),
                "mse_next_latent_vs_cache_training_pair": float(F.mse_loss(eps_next, target).item()),
            }
            if step_idx + 1 < target_eps.shape[0]:
                next_target = target_eps[step_idx + 1].unsqueeze(0)
                row["mse_next_latent_vs_next_cache"] = float(F.mse_loss(eps_next, next_target).item())
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows checked under {args.root_dir}")

    keys = [
        "mse_current_latent_vs_cache",
        "mse_next_latent_vs_cache_training_pair",
        "mse_next_latent_vs_next_cache",
    ]
    print("Per-step checks:")
    for row in rows:
        fields = [f"sample={row['sample']}", f"step={row['step_idx']}", f"t={row['timestep']}"]
        for key in keys:
            if key in row:
                fields.append(f"{key}={row[key]:.8g}")
        print("  " + "  ".join(fields))

    print("\nAverages:")
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        if vals:
            print(f"  {key}: {sum(vals) / len(vals):.8g}")

    print("\nInterpretation:")
    print("  mse_current_latent_vs_cache should be ~0 if target_eps was cached from trajectory[i].")
    print("  mse_next_latent_vs_cache_training_pair is the actual current training pair.")
    print("  If the training-pair MSE is much larger, the inversion target is hard or mispaired.")


if __name__ == "__main__":
    main()
