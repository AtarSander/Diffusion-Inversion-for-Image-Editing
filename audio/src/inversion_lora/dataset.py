# ABOUTME: Dataset over cached AudioLDM2 DDIM trajectories yielding shifted-denoiser training
# ABOUTME: pairs (cleaner latent, timestep) -> frozen-teacher epsilon, with T5-padding collate.

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

CONDITIONING_KEYS = ("generated_prompt_embeds", "t5_prompt_embeds", "t5_attention_mask")


class AudioLDM2TrajectoryDataset(Dataset):
    """Flat view over every DDIM transition in a cached trajectory dataset.

    Item `i` of trajectory `s` is the pair the shifted-denoiser loss needs: the student sees
    the *cleaner* latent `trajectory[i + 1]` at timestep `timesteps[i]`, and must predict the
    frozen teacher's epsilon at the *noisier* `trajectory[i]`, which is `target_eps[i]`.
    """

    def __init__(self, root_dir: str | Path, mmap: bool = True):
        """Index the dataset without loading any latents.

        Args:
            root_dir: Directory containing `sample_*` subdirectories.
            mmap: Memory-map trajectory/target files so a single transition reads only its own
                pages instead of the whole multi-megabyte trajectory.
        """
        self.root_dir = Path(root_dir)
        self.mmap = mmap
        self.samples: list[dict[str, Any]] = []
        self.cumulative_lengths: list[int] = []

        for sample_dir in sorted(self.root_dir.glob("sample_*")):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Incomplete sample (no meta.json): {sample_dir}. Run "
                    "src/inversion_lora/verify_trajectories.py to find every partial sample."
                )
            meta = json.loads(meta_path.read_text())
            num_transitions = int(meta["num_transitions"])
            if num_transitions <= 0:
                continue

            with (sample_dir / "timesteps.json").open("r", encoding="utf-8") as f:
                timesteps = json.load(f)
            if len(timesteps) != num_transitions:
                raise ValueError(
                    f"{sample_dir}: {len(timesteps)} timesteps vs {num_transitions} transitions"
                )

            self.samples.append(
                {
                    "trajectory_path": sample_dir / "latents/trajectory.pt",
                    "target_eps_path": sample_dir / "targets/target_eps.pt",
                    "conditioning_path": sample_dir / "conditioning.pt",
                    "timesteps": timesteps,
                    "num_transitions": num_transitions,
                    "sample_idx": int(meta["sample_idx"]),
                }
            )
            total = num_transitions + (
                self.cumulative_lengths[-1] if self.cumulative_lengths else 0
            )
            self.cumulative_lengths.append(total)

        if not self.samples:
            raise FileNotFoundError(f"No usable sample_* directories in {self.root_dir}")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _locate(self, idx: int) -> tuple[dict[str, Any], int]:
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} out of range for {len(self)} transitions")
        sample_pos = bisect_right(self.cumulative_lengths, idx)
        start = 0 if sample_pos == 0 else self.cumulative_lengths[sample_pos - 1]
        return self.samples[sample_pos], idx - start

    def _load(self, path: Path) -> torch.Tensor:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=self.mmap)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample, step_idx = self._locate(idx)

        trajectory = self._load(sample["trajectory_path"])
        target_eps = self._load(sample["target_eps_path"])
        conditioning = torch.load(
            sample["conditioning_path"], map_location="cpu", weights_only=True
        )

        # .clone() materialises just this slice; without it the mmap of the whole file stays alive.
        x_clean = trajectory[step_idx + 1].clone()
        eps = target_eps[step_idx].clone()
        assert x_clean.shape == eps.shape, (x_clean.shape, eps.shape)

        missing = [key for key in CONDITIONING_KEYS if key not in conditioning]
        if missing:
            raise KeyError(f"{sample['conditioning_path']}: missing conditioning {missing}")

        return {
            "x_clean": x_clean,
            "target_eps": eps,
            "timestep": torch.tensor(sample["timesteps"][step_idx], dtype=torch.long),
            "generated_prompt_embeds": conditioning["generated_prompt_embeds"],
            "t5_prompt_embeds": conditioning["t5_prompt_embeds"],
            "t5_attention_mask": conditioning["t5_attention_mask"],
            "sample_idx": sample["sample_idx"],
            "step_idx": step_idx,
        }


def collate_trajectory_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate transitions, right-padding the variable-length T5 conditioning to the batch max.

    AudioLDM2's T5 stream has a prompt-dependent sequence length, so default collation fails.
    Padding embeddings with zeros and the mask with zeros is exact: the UNet turns the mask into
    an additive -10000 bias, so padded keys contribute nothing to cross-attention.
    """
    max_len = max(item["t5_prompt_embeds"].shape[0] for item in items)
    hidden_dim = items[0]["t5_prompt_embeds"].shape[1]

    t5_embeds = torch.zeros(len(items), max_len, hidden_dim, dtype=items[0]["t5_prompt_embeds"].dtype)
    t5_mask = torch.zeros(len(items), max_len, dtype=items[0]["t5_attention_mask"].dtype)
    for i, item in enumerate(items):
        length = item["t5_prompt_embeds"].shape[0]
        assert item["t5_attention_mask"].shape[0] == length, (
            f"t5 mask length {item['t5_attention_mask'].shape[0]} != embeds length {length}"
        )
        t5_embeds[i, :length] = item["t5_prompt_embeds"]
        t5_mask[i, :length] = item["t5_attention_mask"]

    batch = {
        "x_clean": torch.stack([item["x_clean"] for item in items]),
        "target_eps": torch.stack([item["target_eps"] for item in items]),
        "timestep": torch.stack([item["timestep"] for item in items]),
        "generated_prompt_embeds": torch.stack(
            [item["generated_prompt_embeds"] for item in items]
        ),
        "t5_prompt_embeds": t5_embeds,
        "t5_attention_mask": t5_mask,
        "sample_idx": [item["sample_idx"] for item in items],
        "step_idx": [item["step_idx"] for item in items],
    }
    assert batch["x_clean"].shape == batch["target_eps"].shape
    assert batch["timestep"].shape[0] == len(items)
    return batch
