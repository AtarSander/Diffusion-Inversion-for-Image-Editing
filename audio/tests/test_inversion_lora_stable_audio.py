# ABOUTME: Unit tests for the Stable Audio inversion-LoRA data path: single-tensor conditioning
# ABOUTME: read from the cache, and a collate that stacks it without padding.

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_stable_audio_batch,
)
from src.inversion_lora.train_stable_audio import (  # noqa: E402
    STABLE_AUDIO_CONDITIONING_KEYS,
)

LATENT_SHAPE = (64, 8)
TEXT_AUDIO_SHAPE = (130, 6)


def write_sample(root: Path, sample_idx: int, num_transitions: int) -> None:
    """Write one synthetic Stable Audio trajectory whose latents encode their own index."""
    sample_dir = root / f"sample_{sample_idx:06d}"
    (sample_dir / "latents").mkdir(parents=True)
    (sample_dir / "targets").mkdir(parents=True)

    trajectory = torch.stack(
        [torch.full(LATENT_SHAPE, float(i)) for i in range(num_transitions + 1)]
    )
    target_eps = torch.stack(
        [torch.full(LATENT_SHAPE, -(i + 1) * 10.0) for i in range(num_transitions)]
    )
    timesteps = [999.0 - 50.0 * i for i in range(num_transitions)]

    torch.save(trajectory, sample_dir / "latents/trajectory.pt")
    torch.save(target_eps, sample_dir / "targets/target_eps.pt")
    torch.save(
        {"text_audio": torch.full(TEXT_AUDIO_SHAPE, float(sample_idx))},
        sample_dir / "conditioning.pt",
    )
    (sample_dir / "timesteps.json").write_text(json.dumps(timesteps))
    (sample_dir / "prompt.json").write_text(json.dumps({"prompt": f"p{sample_idx}"}))
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "sample_idx": sample_idx,
                "num_transitions": num_transitions,
                "latent_shape": list(LATENT_SHAPE),
                "text_audio_shape": list(TEXT_AUDIO_SHAPE),
                "model_id": "test",
                "schedule": "beta",
                "num_inference_steps": num_transitions,
            }
        )
    )


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    write_sample(tmp_path, 0, num_transitions=3)
    write_sample(tmp_path, 1, num_transitions=5)
    return tmp_path


def dataset(root: Path) -> AudioLDM2TrajectoryDataset:
    return AudioLDM2TrajectoryDataset(
        root, conditioning_keys=STABLE_AUDIO_CONDITIONING_KEYS
    )


def test_items_carry_only_stable_audio_conditioning(dataset_root: Path):
    item = dataset(dataset_root)[0]
    assert item["text_audio"].shape == TEXT_AUDIO_SHAPE
    assert "t5_prompt_embeds" not in item


def test_pairs_are_shifted_by_one_step(dataset_root: Path):
    """Student input must be trajectory[i+1] while the target is the prediction at trajectory[i]."""
    ds = dataset(dataset_root)
    for idx in range(len(ds)):
        item = ds[idx]
        step = item["step_idx"]
        assert item["x_clean"].unique().tolist() == [float(step + 1)]
        assert item["target_eps"].unique().tolist() == [-(step + 1) * 10.0]


def test_audioldm2_keys_are_rejected_on_a_stable_audio_cache(dataset_root: Path):
    """A cache missing the requested conditioning must fail loudly, not silently drop it."""
    with pytest.raises(KeyError, match="t5_prompt_embeds"):
        AudioLDM2TrajectoryDataset(dataset_root)[0]


def test_collate_stacks_conditioning_in_order(dataset_root: Path):
    ds = dataset(dataset_root)
    items = [ds[0], ds[len(ds) - 1]]
    batch = collate_stable_audio_batch(items)
    assert batch["x_clean"].shape == (2, *LATENT_SHAPE)
    assert batch["text_audio"].shape == (2, *TEXT_AUDIO_SHAPE)
    assert batch["timestep"].tolist() == [item["timestep"].item() for item in items]
    assert batch["sample_idx"] == [item["sample_idx"] for item in items]
    for position, item in enumerate(items):
        assert torch.equal(batch["text_audio"][position], item["text_audio"])
