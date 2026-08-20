# ABOUTME: Unit tests for the reconstruction fixture batching and noise-window cropping that let
# ABOUTME: real-audio latents keep each track's own length.

import sys
from pathlib import Path

import pytest
import torch

AUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.reconstruct import batch_latents, crop_to_window  # noqa: E402


def latent(height: int, examples: int = 1) -> torch.Tensor:
    return torch.zeros(examples, 8, height, 16)


def test_equal_shapes_fill_batches_up_to_the_limit():
    batches = batch_latents([latent(256) for _ in range(10)], batch_size=4)
    assert [b.shape[0] for b in batches] == [4, 4, 2]


def test_distinct_shapes_never_share_a_batch():
    batches = batch_latents([latent(325), latent(1500), latent(437)], batch_size=8)
    assert [tuple(b.shape) for b in batches] == [
        (1, 8, 325, 16),
        (1, 8, 1500, 16),
        (1, 8, 437, 16),
    ]


def test_batching_preserves_order_and_count():
    heights = [256, 256, 600, 600, 600, 128]
    batches = batch_latents([latent(h) for h in heights], batch_size=2)
    assert sum(b.shape[0] for b in batches) == len(heights)
    assert [b.shape[2] for b in batches for _ in range(b.shape[0])] == heights


def test_a_single_batch_is_returned_whole():
    assert [tuple(b.shape) for b in batch_latents([latent(256, examples=3)], 8)] == [
        (3, 8, 256, 16)
    ]


def test_crop_to_window_stacks_mixed_lengths():
    stacked = crop_to_window([latent(325), latent(1500), latent(437)], height=256)
    assert stacked.shape == (3, 8, 256, 16)


def test_crop_to_window_keeps_the_leading_frames():
    tall = torch.arange(6 * 16, dtype=torch.float32).reshape(1, 1, 6, 16)
    assert torch.equal(crop_to_window([tall], height=2), tall[:, :, :2, :])


def test_crop_to_window_rejects_a_latent_shorter_than_the_window():
    with pytest.raises(AssertionError, match="shorter than the 256-frame noise window"):
        crop_to_window([latent(256), latent(200)], height=256)
