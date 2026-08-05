from itertools import accumulate

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset


def _dataset_with_transition_counts(*counts: int) -> LatentTrajectoryDataset:
    dataset = object.__new__(LatentTrajectoryDataset)
    dataset.samples = [{"num_transitions": count} for count in counts]
    dataset.cumulative_lengths = list(accumulate(counts))
    return dataset


def test_final_tail_weights_assign_requested_probability_mass() -> None:
    dataset = _dataset_with_transition_counts(10, 5)
    weights, tail_count, other_count = dataset.final_tail_sampling_weights(
        final_step_fraction=0.2,
        target_draw_fraction=0.5,
    )

    assert (tail_count, other_count) == (3, 12)
    tail_indices = torch.tensor([8, 9, 14])
    tail_mass = weights[tail_indices].sum()
    other_mass = weights.sum() - tail_mass
    assert torch.isclose(tail_mass, torch.tensor(0.5, dtype=torch.double))
    assert torch.isclose(other_mass, torch.tensor(0.5, dtype=torch.double))


def test_weighted_sampler_draws_final_tail_at_requested_rate() -> None:
    dataset = _dataset_with_transition_counts(10, 5)
    weights, _, _ = dataset.final_tail_sampling_weights(0.2, 0.5)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=20_000,
        replacement=True,
        generator=torch.Generator().manual_seed(1234),
    )
    tail_indices = {8, 9, 14}
    tail_draw_fraction = sum(index in tail_indices for index in sampler) / 20_000

    assert tail_draw_fraction == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    ("final_step_fraction", "target_draw_fraction"),
    [(0.0, 0.5), (1.0, 0.5), (0.1, 0.0), (0.1, 1.0)],
)
def test_final_tail_weights_reject_invalid_fractions(
    final_step_fraction: float,
    target_draw_fraction: float,
) -> None:
    dataset = _dataset_with_transition_counts(10, 5)

    with pytest.raises(ValueError):
        dataset.final_tail_sampling_weights(final_step_fraction, target_draw_fraction)


def test_final_tail_weights_reject_dataset_without_non_tail_transitions() -> None:
    dataset = _dataset_with_transition_counts(1)

    with pytest.raises(ValueError, match="non-tail"):
        dataset.final_tail_sampling_weights(0.1, 0.5)
