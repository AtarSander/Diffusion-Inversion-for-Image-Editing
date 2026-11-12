# ABOUTME: Unit tests for steered_tokens: plain addition vs per-token norm-preserving
# ABOUTME: renormalization over the channel dimension.
"""Tests for diff_inversion.steering.hooks.steered_tokens."""

import torch

from diff_inversion.steering.hooks import steered_tokens


def test_plain_addition():
    conditional = torch.randn(1, 7, 16)
    delta = torch.randn(16)
    out = steered_tokens(conditional, delta, renorm=False)
    assert torch.allclose(out, conditional + delta, atol=1e-6)


def test_renorm_preserves_per_token_norms():
    conditional = torch.randn(1, 7, 16) * torch.rand(1, 7, 1) * 10
    delta = torch.randn(16) * 5
    out = steered_tokens(conditional, delta, renorm=True)
    assert out.shape == conditional.shape
    assert torch.allclose(
        out.norm(dim=-1), conditional.norm(dim=-1), rtol=1e-4, atol=1e-5
    )


def test_renorm_moves_direction_toward_delta():
    conditional = torch.randn(1, 7, 16)
    delta = 100.0 * torch.ones(16)
    out = steered_tokens(conditional, delta, renorm=True)
    unit = delta / delta.norm()
    before = (torch.nn.functional.normalize(conditional, dim=-1) @ unit).mean()
    after = (torch.nn.functional.normalize(out, dim=-1) @ unit).mean()
    assert after > before


def test_zero_norm_token_is_finite():
    conditional = torch.zeros(1, 3, 16)
    delta = torch.randn(16)
    out = steered_tokens(conditional, delta, renorm=True)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=-1), torch.zeros(1, 3), atol=1e-6)
