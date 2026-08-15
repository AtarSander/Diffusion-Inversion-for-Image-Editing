"""Dataset of FLUX inversion queries paired with sampling-teacher velocities.

This dataset, its validation-subset rules, and its step-balanced sampler are
project-original. They implement the data contract produced by
``generate_flux_inversion_data``; no external dataset implementation is vendored.
See ``docs/flow_matching_provenance.md``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

REQUIRED_SAMPLE_FILES = (
    "conditioning.pt",
    "latent_image_ids.pt",
    "timesteps.pt",
    "inversion_inputs.pt",
    "target_velocities.pt",
    "meta.json",
)


def _load_torch(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


class FluxInversionDataset(Dataset):
    """Expose every selected trajectory transition as one LoRA training item.

    For sampling transition ``x_i -> x_{i+1}``, the input is the cleaner
    ``x_{i+1}`` queried by naive inversion and the target is the teacher
    velocity originally evaluated at ``x_i``.
    """

    def __init__(
        self,
        root_dir: str | Path | Sequence[str | Path],
        *,
        min_inversion_step: int = 0,
        max_inversion_step: int | None = None,
        max_samples: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        if isinstance(root_dir, (str, Path)):
            root_dirs = [Path(root_dir)]
        else:
            root_dirs = [Path(path) for path in root_dir]
        if not root_dirs:
            raise ValueError("FluxInversionDataset requires at least one root directory.")
        if min_inversion_step < 0:
            raise ValueError("min_inversion_step must be non-negative.")
        if max_inversion_step is not None and max_inversion_step < min_inversion_step:
            raise ValueError("max_inversion_step must be >= min_inversion_step.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")

        self.samples: list[dict[str, Any]] = []
        self.items: list[tuple[int, int]] = []

        sample_dirs = [
            sample_dir
            for root in root_dirs
            for sample_dir in sorted(root.glob("sample_*"))
            if all((sample_dir / name).exists() for name in REQUIRED_SAMPLE_FILES)
        ]
        if max_samples is not None and len(sample_dirs) > max_samples:
            sample_dirs = sorted(
                random.Random(sample_seed).sample(sample_dirs, max_samples),
            )

        for sample_dir in sample_dirs:
            meta = self._load_json(sample_dir / "meta.json")
            num_steps = int(meta.get("num_inference_steps", 0))
            if num_steps <= 0:
                inputs = _load_torch(sample_dir / "inversion_inputs.pt", mmap=True)
                num_steps = int(inputs.shape[0])

            sample_index = len(self.samples)
            self.samples.append(
                {
                    "dir": sample_dir,
                    "meta": meta,
                    "num_steps": num_steps,
                }
            )
            for sampling_step in range(num_steps):
                inversion_step = num_steps - 1 - sampling_step
                if inversion_step < min_inversion_step:
                    continue
                if max_inversion_step is not None and inversion_step > max_inversion_step:
                    continue
                self.items.append((sample_index, sampling_step))

        if not self.samples:
            roots = ", ".join(str(path) for path in root_dirs)
            raise FileNotFoundError(f"No complete FLUX inversion samples found under: {roots}")
        if not self.items:
            raise ValueError("The inversion-step filter removed every dataset item.")

        self._validate_shapes()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, sampling_step = self.items[index]
        sample = self.samples[sample_index]
        sample_dir: Path = sample["dir"]
        num_steps = int(sample["num_steps"])

        inversion_inputs = _load_torch(sample_dir / "inversion_inputs.pt", mmap=True)
        target_velocities = _load_torch(sample_dir / "target_velocities.pt", mmap=True)
        timesteps = _load_torch(sample_dir / "timesteps.pt", mmap=True)
        conditioning = _load_torch(sample_dir / "conditioning.pt")
        latent_image_ids = _load_torch(sample_dir / "latent_image_ids.pt", mmap=True)

        if not isinstance(conditioning, dict):
            raise TypeError(f"Expected conditioning dict in {sample_dir / 'conditioning.pt'}.")

        meta = sample["meta"]
        item = {
            "inversion_input": inversion_inputs[sampling_step],
            "target_velocity": target_velocities[sampling_step],
            "timestep": timesteps[sampling_step].to(dtype=torch.float32),
            "prompt_embeds": self._remove_single_batch(conditioning["prompt_embeds"]),
            "pooled_prompt_embeds": self._remove_single_batch(
                conditioning["pooled_prompt_embeds"]
            ),
            "text_ids": self._remove_single_batch(conditioning["text_ids"]),
            "latent_image_ids": self._remove_single_batch(latent_image_ids),
            "guidance_scale": torch.tensor(
                float(meta["guidance_scale"]),
                dtype=torch.float32,
            ),
            "sample_index": torch.tensor(
                int(meta.get("sample_index", sample_index)),
                dtype=torch.long,
            ),
            "sampling_step": torch.tensor(sampling_step, dtype=torch.long),
            "inversion_step": torch.tensor(
                num_steps - 1 - sampling_step,
                dtype=torch.long,
            ),
        }
        return item

    def _validate_shapes(self) -> None:
        expected_shapes: dict[str, tuple[int, ...]] | None = None
        for sample in self.samples:
            sample_dir: Path = sample["dir"]
            num_steps = int(sample["num_steps"])
            inputs = _load_torch(sample_dir / "inversion_inputs.pt", mmap=True)
            targets = _load_torch(sample_dir / "target_velocities.pt", mmap=True)
            timesteps = _load_torch(sample_dir / "timesteps.pt", mmap=True)
            if not all(isinstance(value, torch.Tensor) for value in (inputs, targets, timesteps)):
                raise TypeError(f"Training arrays in {sample_dir} must be tensors.")
            if inputs.ndim != 3 or targets.shape != inputs.shape:
                raise ValueError(
                    f"Expected matching [steps, tokens, channels] tensors in {sample_dir}, "
                    f"got {tuple(inputs.shape)} and {tuple(targets.shape)}."
                )
            if inputs.shape[0] != num_steps or timesteps.shape != (num_steps,):
                raise ValueError(
                    f"Inconsistent step count in {sample_dir}: meta={num_steps}, "
                    f"inputs={inputs.shape[0]}, timesteps={tuple(timesteps.shape)}."
                )

            conditioning = _load_torch(sample_dir / "conditioning.pt")
            required_keys = {
                "prompt_embeds",
                "pooled_prompt_embeds",
                "text_ids",
            }
            if not isinstance(conditioning, dict) or not required_keys.issubset(conditioning):
                missing = required_keys.difference(
                    conditioning.keys() if isinstance(conditioning, dict) else ()
                )
                raise KeyError(f"Missing conditioning tensors {sorted(missing)} in {sample_dir}.")

            image_ids = self._remove_single_batch(
                _load_torch(sample_dir / "latent_image_ids.pt", mmap=True)
            )
            text_ids = self._remove_single_batch(conditioning["text_ids"])
            current_shapes = {
                "latent": tuple(inputs.shape[1:]),
                "image_ids": tuple(image_ids.shape),
                "text_ids": tuple(text_ids.shape),
                "prompt_embeds": tuple(
                    self._remove_single_batch(conditioning["prompt_embeds"]).shape
                ),
                "pooled_prompt_embeds": tuple(
                    self._remove_single_batch(conditioning["pooled_prompt_embeds"]).shape
                ),
            }
            if expected_shapes is None:
                expected_shapes = current_shapes
            elif current_shapes != expected_shapes:
                raise ValueError(
                    "All FLUX training samples in one dataset must share resolution and "
                    f"conditioning shapes; expected {expected_shapes}, got {current_shapes} "
                    f"in {sample_dir}."
                )

    @staticmethod
    def _remove_single_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 0 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {path}.")
        return value


class BalancedInversionStepSampler(Sampler[int]):
    """Visit every item once while interleaving inversion-step strata.

    A flattened trajectory dataset is balanced over a complete epoch, but an
    ordinary random sampler can still produce optimizer batches dominated by a
    few timesteps. This sampler shuffles items within each timestep and emits a
    shuffled round-robin over timesteps. Consequently, each consecutive window
    of ``number_of_timesteps`` items contains one example from every available
    timestep whenever all strata have the same size.
    """

    def __init__(self, dataset: FluxInversionDataset, *, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_step: dict[int, list[int]] = {}
        for item_index, (sample_index, sampling_step) in enumerate(dataset.items):
            num_steps = int(dataset.samples[sample_index]["num_steps"])
            inversion_step = num_steps - 1 - sampling_step
            self.indices_by_step.setdefault(inversion_step, []).append(item_index)
        if not self.indices_by_step:
            raise ValueError("BalancedInversionStepSampler received an empty dataset.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        step_keys = sorted(self.indices_by_step)
        shuffled_by_step: dict[int, list[int]] = {}
        for inversion_step in step_keys:
            indices = self.indices_by_step[inversion_step]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled_by_step[inversion_step] = [indices[index] for index in order]

        positions = {inversion_step: 0 for inversion_step in step_keys}
        emitted = 0
        while emitted < len(self):
            step_order = torch.randperm(len(step_keys), generator=generator).tolist()
            for step_position in step_order:
                inversion_step = step_keys[step_position]
                position = positions[inversion_step]
                candidates = shuffled_by_step[inversion_step]
                if position >= len(candidates):
                    continue
                yield candidates[position]
                positions[inversion_step] = position + 1
                emitted += 1


def collate_flux_inversion_batch(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot collate an empty FLUX inversion batch.")

    text_ids = items[0]["text_ids"]
    latent_image_ids = items[0]["latent_image_ids"]
    for item in items[1:]:
        if not torch.equal(item["text_ids"], text_ids):
            raise ValueError("FLUX text position IDs must match inside a batch.")
        if not torch.equal(item["latent_image_ids"], latent_image_ids):
            raise ValueError("FLUX image position IDs must match inside a batch.")

    stacked_keys = (
        "inversion_input",
        "target_velocity",
        "timestep",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "guidance_scale",
        "sample_index",
        "sampling_step",
        "inversion_step",
    )
    batch = {key: torch.stack([item[key] for item in items]) for key in stacked_keys}
    batch["text_ids"] = text_ids
    batch["latent_image_ids"] = latent_image_ids
    return batch
