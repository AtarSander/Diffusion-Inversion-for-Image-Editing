import pytest
import torch

from diff_inversion.eval.editor_controls import (
    pix2pix_attention_alignment_loss,
    pnp_reference_embedding_index,
)


def test_pix2pix_legacy_attention_loss_matches_inherited_formula():
    current_a = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 2.0], [4.0, 6.0]]],
        requires_grad=True,
    )
    current_b = torch.tensor([[[3.0, 2.0, 1.0]]], requires_grad=True)
    reference_a = torch.zeros_like(current_a)
    reference_b = torch.ones_like(current_b)

    actual = pix2pix_attention_alignment_loss(
        [(current_a, reference_a), (current_b, reference_b)],
        reduction="legacy_sum",
    )
    expected = (
        (current_a - reference_a).square().sum((1, 2)).mean(0)
        + (current_b - reference_b).square().sum((1, 2)).mean(0)
    )

    torch.testing.assert_close(actual, expected)


def test_pix2pix_per_layer_mean_is_size_and_layer_normalized():
    current_small = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    reference_small = torch.zeros_like(current_small)
    current_large = torch.full((2, 4, 8), 2.0, requires_grad=True)
    reference_large = torch.zeros_like(current_large)

    actual = pix2pix_attention_alignment_loss(
        [(current_small, reference_small), (current_large, reference_large)],
        reduction="per_layer_mean",
    )
    expected = (
        (current_small - reference_small).float().square().mean()
        + (current_large - reference_large).float().square().mean()
    ) / 2

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert current_small.grad is not None
    assert current_large.grad is not None


def test_pix2pix_attention_loss_rejects_invalid_or_empty_inputs():
    with pytest.raises(ValueError, match="Unsupported"):
        pix2pix_attention_alignment_loss([], reduction="not-a-reduction")
    with pytest.raises(RuntimeError, match="did not expose"):
        pix2pix_attention_alignment_loss([], reduction="per_layer_mean")


def test_pnp_reference_embedding_index():
    assert pnp_reference_embedding_index("empty") == 0
    assert pnp_reference_embedding_index("source") == 1
    with pytest.raises(ValueError, match="Unsupported"):
        pnp_reference_embedding_index("target")
