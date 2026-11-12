"""Short CPU-only smoke test for deterministic source-mixture transition sampling."""

from itertools import accumulate

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.train import build_train_sampler


def main() -> None:
    dataset = object.__new__(LatentTrajectoryDataset)
    dataset.samples = [
        {"source_name": "recap_coco", "source_root": "/recap", "num_transitions": 50},
        {"source_name": "diffusiondb_artificial", "source_root": "/artificial", "num_transitions": 50},
    ]
    dataset.cumulative_lengths = list(accumulate(sample["num_transitions"] for sample in dataset.samples))
    sampler = build_train_sampler(
        dataset,
        {
            "mode": "source_mixture",
            "source_weights": {"recap_coco": 0.8, "diffusiondb_artificial": 0.2},
            "num_samples": 1_000,
            "replacement": True,
            "seed": 42,
        },
    )
    assert sampler is not None
    draws = list(sampler)
    artificial_fraction = sum(index >= 50 for index in draws) / len(draws)
    assert len(draws) == 1_000
    assert abs(artificial_fraction - 0.2) < 0.05
    print(f"source-mixture CPU smoke passed; artificial draw fraction={artificial_fraction:.3f}")


if __name__ == "__main__":
    main()
