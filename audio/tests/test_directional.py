# ABOUTME: Known-input tests for the directional CLAP/MuLan score, including the property that
# ABOUTME: makes it worth reporting: an unedited copy of the input scores 0, not high.

import pytest
import torch

from editing.eval_medley import directional_similarity


def test_aligned_directions_score_one():
    """Audio moving exactly the way the caption moves is the maximum score."""
    audio = torch.randn(512)
    text = torch.randn(512)
    delta = torch.randn(512)
    score = directional_similarity(audio, audio + delta, text, text + delta)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_opposite_directions_score_minus_one():
    """Editing away from the target caption is penalised, not merely unrewarded."""
    audio = torch.randn(512)
    text = torch.randn(512)
    delta = torch.randn(512)
    score = directional_similarity(audio, audio - delta, text, text + delta)
    assert score == pytest.approx(-1.0, abs=1e-5)


def test_orthogonal_directions_score_zero():
    audio = torch.zeros(4)
    text = torch.zeros(4)
    audio_moved = torch.tensor([1.0, 0.0, 0.0, 0.0])
    text_moved = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert directional_similarity(audio, audio_moved, text, text_moved) == pytest.approx(0.0)


def test_unedited_copy_scores_zero():
    """The point of the metric: returning the input unchanged earns nothing.

    Plain CLAP/MuLan reward a copy of the input whenever the source already resembles the
    target caption, so they cannot separate "edited correctly" from "did nothing".
    """
    audio = torch.randn(512)
    text = torch.randn(512)
    assert directional_similarity(audio, audio, text, text + torch.randn(512)) == 0.0


def test_score_is_scale_invariant_in_the_audio_delta():
    """Only the direction of the change counts, not how far the edit travelled."""
    audio = torch.randn(512)
    text = torch.randn(512)
    delta_audio = torch.randn(512)
    delta_text = torch.randn(512)
    small = directional_similarity(audio, audio + 0.01 * delta_audio, text, text + delta_text)
    large = directional_similarity(audio, audio + 100.0 * delta_audio, text, text + delta_text)
    assert small == pytest.approx(large, abs=1e-4)


def test_w_scales_the_output():
    audio = torch.randn(512)
    text = torch.randn(512)
    delta = torch.randn(512)
    plain = directional_similarity(audio, audio + delta, text, text + delta)
    scaled = directional_similarity(audio, audio + delta, text, text + delta, w=2.5)
    assert scaled == pytest.approx(2.5 * plain, abs=1e-5)


def test_batched_embedding_is_rejected():
    """A [1, D] embedding would broadcast silently instead of failing."""
    with pytest.raises(AssertionError, match=r"src_audio_emb must be \[D\]"):
        directional_similarity(
            torch.randn(1, 512), torch.randn(512), torch.randn(512), torch.randn(512)
        )
