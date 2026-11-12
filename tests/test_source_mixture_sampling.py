from __future__ import annotations

import json
from itertools import accumulate

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset


def _dataset(source_counts: dict[str, list[int]]) -> LatentTrajectoryDataset:
    dataset = object.__new__(LatentTrajectoryDataset)
    dataset.samples = [
        {"source_name": source_name, "source_root": f"/{source_name}", "num_transitions": count}
        for source_name, counts in source_counts.items()
        for count in counts
    ]
    dataset.cumulative_lengths = list(accumulate(sample["num_transitions"] for sample in dataset.samples))
    return dataset


def test_source_mixture_weights_assign_exact_source_mass() -> None:
    dataset = _dataset({"recap_coco": [100, 50], "diffusiondb_artificial": [10]})
    weights, counts, normalized = dataset.source_mixture_sampling_weights(
        {"recap_coco": 0.8, "diffusiondb_artificial": 0.2}
    )
    assert counts == {"recap_coco": 150, "diffusiondb_artificial": 10}
    assert normalized == {"recap_coco": 0.8, "diffusiondb_artificial": 0.2}
    assert torch.isclose(weights[:150].sum(), torch.tensor(0.8, dtype=torch.double))
    assert torch.isclose(weights[150:].sum(), torch.tensor(0.2, dtype=torch.double))


def test_source_mixture_sampler_is_deterministic_with_requested_draw_count() -> None:
    dataset = _dataset({"recap_coco": [100], "diffusiondb_artificial": [10]})
    weights, _, _ = dataset.source_mixture_sampling_weights(
        {"recap_coco": 0.8, "diffusiondb_artificial": 0.2}
    )
    def draw() -> list[int]:
        return list(
            WeightedRandomSampler(
                weights,
                num_samples=5_000,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            )
        )
    first, second = draw(), draw()
    assert first == second
    assert len(first) == 5_000
    assert sum(index >= 100 for index in first) / len(first) == pytest.approx(0.2, abs=0.02)


def test_source_mixture_requires_exact_source_mapping() -> None:
    dataset = _dataset({"recap_coco": [1], "diffusiondb_artificial": [1]})
    with pytest.raises(ValueError, match="exactly"):
        dataset.source_mixture_sampling_weights({"recap_coco": 1.0})


def test_transition_metadata_exposes_source_without_loading_tensors(tmp_path) -> None:
    roots = []
    for source_name in ("recap_coco", "diffusiondb_artificial"):
        root = tmp_path / source_name
        sample_dir = root / "sample_000000"
        (sample_dir / "latents").mkdir(parents=True)
        torch.save(torch.zeros(3, 1, 4, 4, 4), sample_dir / "latents" / "trajectory.pt")
        (sample_dir / "timesteps.json").write_text("[10, 5, 0]", encoding="utf-8")
        (sample_dir / "meta.json").write_text(
            json.dumps({"sample_idx": 0, "trajectory_length": 3}), encoding="utf-8"
        )
        roots.append(root)

    dataset = LatentTrajectoryDataset(roots, require_training_cache=False, source_names=["recap_coco", "diffusiondb_artificial"])
    assert dataset.transition_metadata(0)["source_name"] == "recap_coco"
    assert dataset.transition_metadata(2)["source_name"] == "diffusiondb_artificial"
    assert dataset.transition_metadata(2)["source_root"] == str(roots[1])
