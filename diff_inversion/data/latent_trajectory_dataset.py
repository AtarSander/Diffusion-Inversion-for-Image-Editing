import json
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class LatentTrajectoryDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path | Sequence[str | Path],
        latents_file_name: str = "trajectory.pt",
        conditioning_file_name: str = "conditioning.pt",
        targets_dir_name: str = "targets",
        target_eps_file_name: str = "target_eps.pt",
        target_uncond_eps_file_name: str = "target_eps_uncond.pt",
        load_cfg_branch_targets: bool = False,
        require_training_cache: bool = True,
    ):
        if isinstance(root_dir, (str, Path)):
            self.root_dirs = [Path(root_dir)]
        else:
            self.root_dirs = [Path(path) for path in root_dir]
        if not self.root_dirs:
            raise ValueError("LatentTrajectoryDataset requires at least one root directory.")
        self.latents_file_name = latents_file_name
        self.conditioning_file_name = conditioning_file_name
        self.targets_dir_name = targets_dir_name
        self.target_eps_file_name = target_eps_file_name
        self.target_uncond_eps_file_name = target_uncond_eps_file_name
        self.load_cfg_branch_targets = bool(load_cfg_branch_targets)
        self.require_training_cache = require_training_cache
        self.samples: list[dict[str, Any]] = []
        self.cumulative_lengths: list[int] = []

        for root_dir in self.root_dirs:
            for sample_dir in sorted(root_dir.glob("sample_*")):
                latents_dir = sample_dir / "latents"
                timesteps_path = sample_dir / "timesteps.json"
                if not latents_dir.exists() or not timesteps_path.exists():
                    continue

                prompt_path = sample_dir / "prompt.json"
                meta = self._load_json(sample_dir / "meta.json")
                conditioning_path = sample_dir / self.conditioning_file_name
                target_eps_path = sample_dir / self.targets_dir_name / self.target_eps_file_name
                target_uncond_eps_path = (
                    sample_dir / self.targets_dir_name / self.target_uncond_eps_file_name
                )
                self._validate_training_cache(
                    sample_dir,
                    conditioning_path,
                    target_eps_path,
                    target_uncond_eps_path,
                )

                trajectory_path = latents_dir / self.latents_file_name
                if trajectory_path.exists():
                    trajectory_length = int(meta.get("trajectory_length", 0))
                    if trajectory_length <= 0:
                        trajectory_length = int(
                            torch.load(trajectory_path, map_location="cpu").shape[0]
                        )
                    self._add_sample(
                        {
                            "sample_dir": sample_dir,
                            "format": "stacked_pt",
                            "trajectory_path": trajectory_path,
                            "trajectory_length": trajectory_length,
                            "timesteps_path": timesteps_path,
                            "prompt_path": prompt_path,
                            "conditioning_path": conditioning_path,
                            "target_eps_path": target_eps_path,
                            "target_uncond_eps_path": target_uncond_eps_path,
                            "guidance_scale": meta.get("guidance_scale"),
                            "sample_idx": meta.get("sample_idx"),
                        }
                    )
                    continue

                latent_paths = sorted(latents_dir.glob("x_*.pt"))
                if len(latent_paths) >= 2:
                    self._add_sample(
                        {
                            "sample_dir": sample_dir,
                            "format": "per_step_pt",
                            "latent_paths": latent_paths,
                            "trajectory_length": len(latent_paths),
                            "timesteps_path": timesteps_path,
                            "prompt_path": prompt_path,
                            "conditioning_path": conditioning_path,
                            "target_eps_path": target_eps_path,
                            "target_uncond_eps_path": target_uncond_eps_path,
                            "guidance_scale": meta.get("guidance_scale"),
                            "sample_idx": meta.get("sample_idx"),
                        }
                    )

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample, step_idx = self._locate(idx)

        if sample["format"] == "stacked_pt":
            trajectory = torch.load(sample["trajectory_path"], map_location="cpu")
            x_clean = trajectory[step_idx + 1]
        else:
            x_clean = torch.load(sample["latent_paths"][step_idx + 1], map_location="cpu")

        timestep = self._transition_timestep(
            self._timesteps(sample),
            step_idx,
            sample["trajectory_length"],
        )
        conditioning = torch.load(sample["conditioning_path"], map_location="cpu")
        target_eps = torch.load(sample["target_eps_path"], map_location="cpu")[step_idx]

        item = {
            "x_clean": self._squeeze_latent(x_clean),
            "timestep": torch.tensor(timestep, dtype=torch.long),
            "prompt_embeds": self._squeeze_batch_dim(conditioning["prompt_embeds"]),
            "target_eps": self._squeeze_latent(target_eps),
            "sample_idx": sample["sample_idx"],
            "step_idx": step_idx,
        }
        if "pooled_prompt_embeds" in conditioning:
            item["pooled_prompt_embeds"] = self._squeeze_batch_dim(
                conditioning["pooled_prompt_embeds"]
            )
        if "add_time_ids" in conditioning:
            item["add_time_ids"] = self._squeeze_batch_dim(conditioning["add_time_ids"])
        if self.load_cfg_branch_targets:
            item.update(self._cfg_branch_item(sample, conditioning, step_idx))
        return item

    def transition_metadata(self, idx: int) -> dict[str, Any]:
        """Return transition identity without loading trajectory or conditioning tensors."""
        sample, step_idx = self._locate(idx)
        num_transitions = int(sample["num_transitions"])
        return {
            "dataset_idx": int(idx),
            "sample_idx": sample["sample_idx"],
            "sample_dir": str(sample["sample_dir"]),
            "step_idx": int(step_idx),
            # Sampling is stored noise -> image, while inversion runs image -> noise.
            "inversion_step": num_transitions - 1 - int(step_idx),
            "num_transitions": num_transitions,
            "timestep": self._transition_timestep(
                self._timesteps(sample),
                step_idx,
                int(sample["trajectory_length"]),
            ),
        }

    def _cfg_branch_item(
        self,
        sample: dict[str, Any],
        conditioning: dict[str, torch.Tensor],
        step_idx: int,
    ) -> dict[str, Any]:
        if "negative_prompt_embeds" not in conditioning:
            raise KeyError(
                f"Missing negative_prompt_embeds in {sample['conditioning_path']} "
                "required for CFG training."
            )

        target_eps_uncond = torch.load(
            sample["target_uncond_eps_path"],
            map_location="cpu",
        )[step_idx]
        item: dict[str, Any] = {
            "negative_prompt_embeds": self._squeeze_batch_dim(
                conditioning["negative_prompt_embeds"]
            ),
            "target_eps_uncond": self._squeeze_latent(target_eps_uncond),
        }
        if "negative_pooled_prompt_embeds" in conditioning:
            item["negative_pooled_prompt_embeds"] = self._squeeze_batch_dim(
                conditioning["negative_pooled_prompt_embeds"]
            )
        elif "pooled_prompt_embeds" in conditioning:
            raise KeyError(
                f"Missing negative_pooled_prompt_embeds in {sample['conditioning_path']} "
                "required for SDXL CFG training."
            )

        if sample.get("guidance_scale") is not None:
            item["sample_guidance_scale"] = torch.tensor(
                float(sample["guidance_scale"]),
                dtype=torch.float32,
            )
        return item

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _squeeze_latent(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4 and latent.shape[0] == 1:
            return latent[0]
        return latent

    @staticmethod
    def _squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 0 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    @staticmethod
    def _transition_timestep(timesteps: list[int], step_idx: int, trajectory_length: int) -> int:
        if len(timesteps) == trajectory_length:
            return int(timesteps[step_idx + 1])
        if len(timesteps) == trajectory_length - 1:
            return int(timesteps[step_idx])
        raise ValueError(
            "Unexpected timestep count for trajectory: "
            f"got {len(timesteps)}, expected {trajectory_length} or {trajectory_length - 1}"
        )

    def _validate_training_cache(
        self,
        sample_dir: Path,
        conditioning_path: Path,
        target_eps_path: Path,
        target_uncond_eps_path: Path,
    ) -> None:
        if not self.require_training_cache:
            return
        required_paths = [conditioning_path, target_eps_path]
        if self.load_cfg_branch_targets:
            required_paths.append(target_uncond_eps_path)
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Missing cached training tensors for "
                f"{sample_dir}: {', '.join(missing_paths)}. "
                "Run diff_inversion/data/precompute_training_cache.py first."
            )

    def _add_sample(self, sample: dict[str, Any]) -> None:
        num_transitions = sample["trajectory_length"] - 1
        if num_transitions <= 0:
            return
        sample["num_transitions"] = num_transitions
        self.samples.append(sample)
        total = (
            num_transitions
            if not self.cumulative_lengths
            else (self.cumulative_lengths[-1] + num_transitions)
        )
        self.cumulative_lengths.append(total)

    def _locate(self, idx: int) -> tuple[dict[str, Any], int]:
        sample_pos = bisect_right(self.cumulative_lengths, idx)
        sample_start = 0 if sample_pos == 0 else self.cumulative_lengths[sample_pos - 1]
        return self.samples[sample_pos], idx - sample_start

    def _timesteps(self, sample: dict[str, Any]) -> list[int]:
        if "timesteps" not in sample:
            with sample["timesteps_path"].open("r", encoding="utf-8") as f:
                sample["timesteps"] = json.load(f)
        return sample["timesteps"]

    def _prompt(self, sample: dict[str, Any]) -> str:
        if "prompt" not in sample:
            sample["prompt"] = self._load_json(sample["prompt_path"]).get("prompt", "")
        return sample["prompt"]
