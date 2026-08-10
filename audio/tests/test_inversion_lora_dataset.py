# ABOUTME: Unit tests for the AudioLDM2 trajectory dataset: flat transition indexing, the
# ABOUTME: shifted-denoiser pairing, and right-padding of variable-length T5 conditioning.

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_trajectory_batch,
)

LATENT_SHAPE = (8, 16, 4)


def write_sample(root: Path, sample_idx: int, num_transitions: int, t5_len: int) -> None:
    """Write one synthetic trajectory sample whose latents encode their own index."""
    sample_dir = root / f"sample_{sample_idx:06d}"
    (sample_dir / "latents").mkdir(parents=True)
    (sample_dir / "targets").mkdir(parents=True)

    # trajectory[i] is filled with value i, so a mis-shifted pair is immediately visible.
    trajectory = torch.stack(
        [torch.full(LATENT_SHAPE, float(i)) for i in range(num_transitions + 1)]
    )
    # target_eps[i] is filled with -(i + 1) * 10, distinct from every latent value.
    target_eps = torch.stack(
        [torch.full(LATENT_SHAPE, -(i + 1) * 10.0) for i in range(num_transitions)]
    )
    timesteps = [999 - 5 * i for i in range(num_transitions)]

    torch.save(trajectory, sample_dir / "latents/trajectory.pt")
    torch.save(target_eps, sample_dir / "targets/target_eps.pt")
    torch.save(
        {
            "generated_prompt_embeds": torch.randn(8, 12),
            "t5_prompt_embeds": torch.randn(t5_len, 6),
            "t5_attention_mask": torch.ones(t5_len, dtype=torch.long),
        },
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
                "t5_seq_len": t5_len,
                "model_id": "test",
                "num_inference_steps": num_transitions,
            }
        )
    )


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    write_sample(tmp_path, 0, num_transitions=4, t5_len=3)
    write_sample(tmp_path, 1, num_transitions=6, t5_len=7)
    return tmp_path


def test_length_is_total_transitions(dataset_root: Path):
    assert len(AudioLDM2TrajectoryDataset(dataset_root)) == 10


def test_pairs_are_shifted_by_one_step(dataset_root: Path):
    """Student input must be trajectory[i+1] while the target is the epsilon at trajectory[i]."""
    ds = AudioLDM2TrajectoryDataset(dataset_root)
    for idx in range(len(ds)):
        item = ds[idx]
        step = item["step_idx"]
        assert item["x_clean"].unique().tolist() == [float(step + 1)], (
            f"idx {idx}: student saw latent {item['x_clean'].unique().tolist()}, want {step + 1}"
        )
        assert item["target_eps"].unique().tolist() == [-(step + 1) * 10.0]
        assert item["timestep"].item() == 999 - 5 * step


def test_flat_index_maps_across_samples(dataset_root: Path):
    ds = AudioLDM2TrajectoryDataset(dataset_root)
    assert [(ds[i]["sample_idx"], ds[i]["step_idx"]) for i in range(5)] == [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
    ]
    assert (ds[9]["sample_idx"], ds[9]["step_idx"]) == (1, 5)


def test_out_of_range_index_raises(dataset_root: Path):
    ds = AudioLDM2TrajectoryDataset(dataset_root)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_incomplete_sample_is_rejected(dataset_root: Path):
    (dataset_root / "sample_000001/meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete sample"):
        AudioLDM2TrajectoryDataset(dataset_root)


def test_collate_right_pads_t5_to_batch_max(dataset_root: Path):
    ds = AudioLDM2TrajectoryDataset(dataset_root)
    batch = collate_trajectory_batch([ds[0], ds[len(ds) - 1]])

    assert batch["t5_prompt_embeds"].shape == (2, 7, 6)
    assert batch["t5_attention_mask"].shape == (2, 7)
    # Shorter item keeps 3 real tokens then zeros; mask marks exactly the real ones.
    assert batch["t5_attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert batch["t5_attention_mask"][1].tolist() == [1] * 7
    assert torch.all(batch["t5_prompt_embeds"][0, 3:] == 0)
    assert torch.equal(batch["t5_prompt_embeds"][0, :3], ds[0]["t5_prompt_embeds"])
    assert batch["x_clean"].shape == (2, *LATENT_SHAPE)
    assert batch["generated_prompt_embeds"].shape == (2, 8, 12)


def test_collate_is_order_preserving(dataset_root: Path):
    ds = AudioLDM2TrajectoryDataset(dataset_root)
    items = [ds[2], ds[7]]
    batch = collate_trajectory_batch(items)
    for i, item in enumerate(items):
        assert torch.equal(batch["x_clean"][i], item["x_clean"])
        assert torch.equal(batch["target_eps"][i], item["target_eps"])
        assert batch["timestep"][i].item() == item["timestep"].item()
