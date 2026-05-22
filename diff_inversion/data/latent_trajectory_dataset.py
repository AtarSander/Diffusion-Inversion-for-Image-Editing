"""Dataset for SDXL latent trajectories saved by ``generate_sdxl_samples``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class LatentTrajectoryDataset(Dataset):
    """Flatten saved sample trajectories into per-denoising-step training items."""

    def __init__(
        self,
        root_dir: str | Path,
        latents_dir_name: str = "latents",
        pred_noises_dir_name: str = "pred_noises",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.latents_dir_name = latents_dir_name
        self.pred_noises_dir_name = pred_noises_dir_name
        self.items: list[dict[str, Any]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Trajectory root does not exist: {self.root_dir}")

        for sample_dir in sorted(self.root_dir.glob("sample_*")):
            self._add_sample(sample_dir)

        if not self.items:
            raise ValueError(f"No complete trajectory items found in {self.root_dir}")

    def _add_sample(self, sample_dir: Path) -> None:
        latents_dir = sample_dir / self.latents_dir_name
        pred_noises_dir = sample_dir / self.pred_noises_dir_name
        timesteps_path = sample_dir / "timesteps.json"
        prompt_path = sample_dir / "prompt.json"

        if not (
            latents_dir.exists()
            and pred_noises_dir.exists()
            and timesteps_path.exists()
            and prompt_path.exists()
        ):
            return

        latent_paths = sorted(latents_dir.glob("x_*.pt"))
        pred_noise_paths = sorted(pred_noises_dir.glob("noise_*.pt"))
        if len(latent_paths) < 2 or len(pred_noise_paths) != len(latent_paths) - 1:
            return

        timesteps = self._read_json(timesteps_path)
        if not isinstance(timesteps, list) or len(timesteps) < len(pred_noise_paths):
            return

        prompt_record = self._read_json(prompt_path)
        prompt = str(prompt_record.get("prompt", "")) if isinstance(prompt_record, dict) else ""
        if not prompt:
            return

        sample_idx = self._sample_idx_from_dir(sample_dir)
        if sample_idx is None:
            return
        num_steps = len(pred_noise_paths)
        for step_idx, noise_path in enumerate(pred_noise_paths):
            # During generation, noise_i is predicted from latent_i before the DDIM step.
            # During inversion, latent_{i+1} is the earlier available state, so this pair
            # teaches the adapter to recover the next-step prediction from that state.
            inversion_step_idx = num_steps - 1 - step_idx
            self.items.append(
                {
                    "input_latent_path": latent_paths[step_idx + 1],
                    "target_eps_path": noise_path,
                    "timestep": int(timesteps[step_idx]),
                    "prompt": prompt,
                    "sample_idx": sample_idx,
                    "step_idx": step_idx,
                    "inversion_step_idx": inversion_step_idx,
                    "num_steps": num_steps,
                }
            )

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _sample_idx_from_dir(sample_dir: Path) -> int | None:
        try:
            return int(sample_dir.name.removeprefix("sample_"))
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        input_latent = torch.load(
            item["input_latent_path"], map_location="cpu", weights_only=False
        )
        target_eps = torch.load(item["target_eps_path"], map_location="cpu", weights_only=False)

        return {
            "input_latent": self._squeeze_saved_batch(input_latent),
            "target_eps": self._squeeze_saved_batch(target_eps),
            "timestep": torch.tensor(item["timestep"], dtype=torch.long),
            "prompt": item["prompt"],
            "sample_idx": item["sample_idx"],
            "step_idx": item["step_idx"],
            "inversion_step_idx": item["inversion_step_idx"],
            "num_steps": item["num_steps"],
        }

    @staticmethod
    def _squeeze_saved_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    def early_inversion_weights(self, early_fraction: float, early_weight: float) -> torch.Tensor:
        """Return sampler weights that favor the first fraction of inversion steps."""
        if not 0 < early_fraction <= 1:
            raise ValueError(f"early_fraction must be in (0, 1], got {early_fraction}")
        if early_weight <= 0:
            raise ValueError(f"early_weight must be positive, got {early_weight}")

        weights = []
        for item in self.items:
            early_steps = max(1, math.ceil(item["num_steps"] * early_fraction))
            weight = early_weight if item["inversion_step_idx"] < early_steps else 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)
