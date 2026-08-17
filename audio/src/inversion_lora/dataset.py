# ABOUTME: Dataset over cached AudioLDM2 DDIM trajectories yielding shifted-denoiser training
# ABOUTME: pairs (cleaner latent, timestep) -> frozen-teacher epsilon, with T5-padding collate.

import json
import random
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

    def __init__(
        self,
        root_dir: str | Path,
        mmap: bool = True,
        sample_ids: set[int] | None = None,
    ):
        """Index the dataset without loading any latents.

        Args:
            root_dir: Directory containing `sample_*` subdirectories.
            mmap: Memory-map trajectory/target files so a single transition reads only its own
                pages instead of the whole multi-megabyte trajectory.
            sample_ids: Restrict to these `sample_idx` values. Splits happen at trajectory
                level so transitions from one trajectory never straddle a train/val boundary.
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
            if sample_ids is not None and int(meta["sample_idx"]) not in sample_ids:
                continue
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


def transitions_below_timestep(
    dataset: AudioLDM2TrajectoryDataset, max_timestep: int
) -> list[int]:
    """Flat indices of the transitions whose timestep is at or below `max_timestep`.

    Reads the cached timesteps only, so restricting training to part of the schedule costs one
    pass over the index rather than any latent I/O.

    Args:
        dataset: An indexed trajectory dataset.
        max_timestep: Noisiest timestep to keep.

    Returns:
        Indices into `dataset`, ascending.
    """
    keep: list[int] = []
    start = 0
    for sample in dataset.samples:
        keep.extend(
            start + i for i, t in enumerate(sample["timesteps"]) if t <= max_timestep
        )
        start += sample["num_transitions"]
    if not keep:
        raise ValueError(f"No transitions at or below timestep {max_timestep}")
    return keep


def split_sample_ids(
    root_dir: str | Path, val_fraction: float, seed: int = 0
) -> tuple[set[int], set[int]]:
    """Split trajectories (not transitions) into train and validation id sets.

    Args:
        root_dir: Dataset directory containing `sample_*` subdirectories.
        val_fraction: Fraction of trajectories held out for validation.
        seed: Seed for the deterministic shuffle.

    Returns:
        `(train_ids, val_ids)`, disjoint and jointly covering every complete sample.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    ids = sorted(
        int(json.loads((d / "meta.json").read_text())["sample_idx"])
        for d in Path(root_dir).glob("sample_*")
        if (d / "meta.json").exists()
    )
    if not ids:
        raise FileNotFoundError(f"No complete samples in {root_dir}")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    num_val = int(round(val_fraction * len(shuffled)))
    if val_fraction > 0 and num_val == 0:
        num_val = 1
    val_ids = set(shuffled[:num_val])
    train_ids = set(shuffled[num_val:])
    if not train_ids:
        raise ValueError(
            f"val_fraction={val_fraction} leaves no training trajectories out of {len(ids)}"
        )
    assert train_ids.isdisjoint(val_ids)
    return train_ids, val_ids


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
