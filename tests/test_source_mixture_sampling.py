import json

import pytest
import torch
from omegaconf import OmegaConf

from diff_inversion.data.latent_trajectory_dataset import LatentTrajectoryDataset
from diff_inversion.modeling.train import build_train_sampler


def _dataset(tmp_path):
    roots = [tmp_path / "recap", tmp_path / "art"]
    for root, steps in zip(roots, [4, 2]):
        sample = root / "sample_000000"
        (sample / "latents").mkdir(parents=True)
        (sample / "latents" / "trajectory.pt").touch()
        (sample / "meta.json").write_text(json.dumps({"trajectory_length": steps + 1}))
        (sample / "timesteps.json").write_text(json.dumps(list(range(steps, -1, -1))))
    return LatentTrajectoryDataset(
        roots, source_names=["recap", "art"], require_training_cache=False
    )


def test_source_mass_and_metadata_without_tensor_loading(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Metadata and sampling must not load tensors")
    monkeypatch.setattr(torch, "load", fail)
    dataset = _dataset(tmp_path)
    weights, counts, masses = dataset.source_mixture_sampling_weights({"recap": 8, "art": 2})
    assert counts == {"recap": 4, "art": 2}
    assert masses == {"recap": 0.8, "art": 0.2}
    assert weights[:4].sum().item() == pytest.approx(0.8)
    assert weights[4:].sum().item() == pytest.approx(0.2)
    assert [dataset.transition_metadata(i)["source_name"] for i in range(6)] == ["recap"] * 4 + ["art"] * 2


def test_sampler_is_deterministic_and_preserves_draw_count(tmp_path):
    dataset = _dataset(tmp_path)
    cfg = OmegaConf.create(dict(mode="source_mixture", source_weights=dict(recap=0.8, art=0.2), num_samples=1000, seed=42))
    draws = list(build_train_sampler(dataset, cfg))
    assert len(draws) == 1000
    assert draws == list(build_train_sampler(dataset, cfg))
    assert sum(i < 4 for i in draws) / len(draws) == pytest.approx(0.8, abs=0.05)


@pytest.mark.parametrize("weights", [{"recap": 1}, {"recap": 0, "art": 0}, {"recap": -1, "art": 2}, {"recap": float("nan"), "art": 1}])
def test_invalid_sources_rejected(tmp_path, weights):
    with pytest.raises(ValueError):
        _dataset(tmp_path).source_mixture_sampling_weights(weights)


def test_source_names_must_match_roots(tmp_path):
    with pytest.raises(ValueError, match="source_names"):
        LatentTrajectoryDataset([tmp_path], source_names=["a", "b"])
