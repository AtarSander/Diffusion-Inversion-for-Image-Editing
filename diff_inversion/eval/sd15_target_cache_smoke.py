"""Hydra entry point for SD1.5 target-cache alignment smoke checks."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.utils import make_pipe


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


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
    raise ValueError(
        "Expected trajectory [T,1,C,H,W] or [T,C,H,W], "
        f"got {tuple(trajectory.shape)}"
    )


def transition_timestep(timesteps: list[int], step_idx: int, trajectory_length: int) -> int:
    if len(timesteps) == trajectory_length:
        return int(timesteps[step_idx + 1])
    if len(timesteps) == trajectory_length - 1:
        return int(timesteps[step_idx])
    raise ValueError(
        f"Unexpected timestep count: got {len(timesteps)}, "
        f"expected {trajectory_length} or {trajectory_length - 1}"
    )


def sample_dirs(root: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in root.glob("sample_*") if path.is_dir())[:max_samples]


@torch.no_grad()
def predict_eps(
    pipe,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    latents = latents.to(device=pipe.device, dtype=pipe.unet.dtype)
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    timesteps = torch.full(
        (latents.shape[0],),
        int(timestep),
        device=pipe.device,
        dtype=torch.long,
    )
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


def run_check(cfg: DictConfig) -> list[dict[str, float | int | str]]:
    root_dir = _resolve_path(cfg.root_dir)
    device = str(cfg.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run this smoke test on a GPU node.")

    pipe = make_pipe(cfg.model, device)
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(int(cfg.model.num_inference_steps), device=device)

    rows = []
    for sample_dir in sample_dirs(root_dir, int(cfg.max_samples)):
        trajectory = squeeze_trajectory(
            load_tensor(sample_dir / str(cfg.latents_dir_name) / str(cfg.latents_file_name))
        )
        target_eps = load_tensor(
            sample_dir / str(cfg.targets_dir_name) / str(cfg.target_eps_file_name)
        ).float()
        conditioning = load_tensor(sample_dir / str(cfg.conditioning_file_name))
        prompt_embeds = conditioning["prompt_embeds"]
        with (sample_dir / "timesteps.json").open("r", encoding="utf-8") as f:
            timesteps = json.load(f)

        num_steps = min(int(cfg.max_steps), int(target_eps.shape[0]), int(trajectory.shape[0]) - 1)
        for step_idx in range(num_steps):
            timestep = transition_timestep(timesteps, step_idx, int(trajectory.shape[0]))
            target = target_eps[step_idx].unsqueeze(0)
            eps_current = predict_eps(pipe, trajectory[step_idx], timestep, prompt_embeds)
            eps_next = predict_eps(pipe, trajectory[step_idx + 1], timestep, prompt_embeds)

            row: dict[str, float | int | str] = {
                "sample": sample_dir.name,
                "step_idx": int(step_idx),
                "timestep": int(timestep),
                "mse_current_latent_vs_cache": float(F.mse_loss(eps_current, target).item()),
                "mse_next_latent_vs_cache_training_pair": float(
                    F.mse_loss(eps_next, target).item()
                ),
            }
            if step_idx + 1 < target_eps.shape[0]:
                next_target = target_eps[step_idx + 1].unsqueeze(0)
                row["mse_next_latent_vs_next_cache"] = float(
                    F.mse_loss(eps_next, next_target).item()
                )
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows checked under {root_dir}")
    return rows


def log_summary(rows: list[dict[str, float | int | str]]) -> None:
    keys = [
        "mse_current_latent_vs_cache",
        "mse_next_latent_vs_cache_training_pair",
        "mse_next_latent_vs_next_cache",
    ]
    logger.info("Per-step checks:")
    for row in rows:
        fields = [f"sample={row['sample']}", f"step={row['step_idx']}", f"t={row['timestep']}"]
        for key in keys:
            if key in row:
                fields.append(f"{key}={float(row[key]):.8g}")
        logger.info("  {}", "  ".join(fields))

    logger.info("Averages:")
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            logger.info("  {}: {:.8g}", key, sum(values) / len(values))

    logger.info(
        "Interpretation: mse_current_latent_vs_cache should be near zero if target_eps "
        "was cached from trajectory[i]; mse_next_latent_vs_cache_training_pair is the "
        "current training pair."
    )


@hydra.main(config_path="../../config", config_name="eval/sd15_target_cache_smoke", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("SD1.5 target-cache smoke config:\n{}", OmegaConf.to_yaml(cfg))
    rows = run_check(cfg)
    log_summary(rows)

    output_path = OmegaConf.select(cfg, "output_json", default=None)
    if output_path:
        path = _resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        logger.info("Wrote smoke results to {}", path)


if __name__ == "__main__":
    main()
