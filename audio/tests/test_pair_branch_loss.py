# ABOUTME: Unit tests for the CFG pair-branch pieces: the guidance combination collapsing at w=1,
# ABOUTME: and the dataset yielding or demanding the cached unconditional branch.

import json
import sys
from pathlib import Path

import pytest
import torch

AUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.dataset import (  # noqa: E402
    AudioLDM2TrajectoryDataset,
    collate_trajectory_batch,
)


def write_sample(root: Path, idx: int, steps: int = 4, with_uncond: bool = True) -> None:
    """Write one minimal trajectory sample directory."""
    d = root / f"sample_{idx:06d}"
    (d / "latents").mkdir(parents=True)
    (d / "targets").mkdir(parents=True)
    torch.save(torch.randn(steps + 1, 8, 16, 16), d / "latents/trajectory.pt")
    torch.save(torch.randn(steps, 8, 16, 16), d / "targets/target_eps.pt")
    if with_uncond:
        torch.save(torch.randn(steps, 8, 16, 16), d / "targets/uncond_eps.pt")
    torch.save(
        {
            "generated_prompt_embeds": torch.randn(8, 768),
            "t5_prompt_embeds": torch.randn(5, 1024),
            "t5_attention_mask": torch.ones(5, dtype=torch.long),
        },
        d / "conditioning.pt",
    )
    (d / "timesteps.json").write_text(json.dumps([999, 750, 500, 250][:steps]))
    (d / "meta.json").write_text(json.dumps({"sample_idx": idx, "num_transitions": steps}))


def test_combination_collapses_to_the_conditional_branch_at_w_one():
    eps_u, eps_c = torch.randn(2, 4), torch.randn(2, 4)
    assert torch.allclose(eps_u + 1.0 * (eps_c - eps_u), eps_c)


def test_combination_amplifies_the_branch_difference():
    eps_u, eps_c = torch.zeros(3), torch.ones(3)
    assert torch.allclose(eps_u + 2.5 * (eps_c - eps_u), torch.full((3,), 2.5))


def test_dataset_yields_uncond_when_asked(tmp_path):
    write_sample(tmp_path, 0)
    item = AudioLDM2TrajectoryDataset(tmp_path, load_uncond=True)[0]
    assert item["uncond_eps"].shape == item["target_eps"].shape


def test_dataset_omits_uncond_by_default(tmp_path):
    write_sample(tmp_path, 0)
    assert "uncond_eps" not in AudioLDM2TrajectoryDataset(tmp_path)[0]


def test_dataset_refuses_load_uncond_without_the_file(tmp_path):
    write_sample(tmp_path, 0, with_uncond=False)
    with pytest.raises(FileNotFoundError, match="save_uncond_target=true"):
        AudioLDM2TrajectoryDataset(tmp_path, load_uncond=True)


def test_collate_carries_uncond_through(tmp_path):
    write_sample(tmp_path, 0)
    dataset = AudioLDM2TrajectoryDataset(tmp_path, load_uncond=True)
    batch = collate_trajectory_batch([dataset[0], dataset[1]])
    assert batch["uncond_eps"].shape == batch["target_eps"].shape


def test_collate_omits_uncond_when_absent(tmp_path):
    write_sample(tmp_path, 0)
    dataset = AudioLDM2TrajectoryDataset(tmp_path)
    assert "uncond_eps" not in collate_trajectory_batch([dataset[0], dataset[1]])
