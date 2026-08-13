# ABOUTME: Known-input tests for the LoRA EMA: decay semantics, convergence to a held value, and
# ABOUTME: that the shadow is a copy rather than a view of the live parameters.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.train import LoRAEMA  # noqa: E402


def parameters(value: float) -> dict[str, torch.Tensor]:
    return {"lora_A": torch.full((2, 3), value), "lora_B": torch.full((3, 2), value)}


def test_one_update_is_a_plain_interpolation():
    ema = LoRAEMA(parameters(0.0), decay=0.9)
    ema.update(parameters(1.0))
    for tensor in ema.state_dict().values():
        assert torch.allclose(tensor, torch.full_like(tensor, 0.1))


def test_decay_zero_tracks_the_live_weights_exactly():
    ema = LoRAEMA(parameters(0.0), decay=0.0)
    ema.update(parameters(7.0))
    for tensor in ema.state_dict().values():
        assert torch.allclose(tensor, torch.full_like(tensor, 7.0))


def test_repeated_updates_converge_to_the_held_value():
    """Held constant, the average must approach it, not stall partway."""
    ema = LoRAEMA(parameters(0.0), decay=0.9)
    for _ in range(200):
        ema.update(parameters(1.0))
    for tensor in ema.state_dict().values():
        assert torch.allclose(tensor, torch.ones_like(tensor), atol=1e-6)


def test_shadow_lags_a_moving_parameter():
    """The point of EMA: it must not equal the latest value after a jump."""
    ema = LoRAEMA(parameters(0.0), decay=0.99)
    ema.update(parameters(1.0))
    shadow = ema.state_dict()["lora_A"]
    assert shadow.max() < 0.5, "decay=0.99 should barely move after a single step"


def test_shadow_is_not_a_view_of_the_parameters():
    """A view would make the 'average' track the live weights exactly and silently."""
    live = parameters(1.0)
    ema = LoRAEMA(live, decay=0.9)
    live["lora_A"].mul_(100.0)
    assert torch.allclose(ema.state_dict()["lora_A"], torch.ones(2, 3))


def test_invalid_decay_is_rejected():
    with pytest.raises(AssertionError, match="decay must be in"):
        LoRAEMA(parameters(0.0), decay=1.0)


def test_load_state_dict_round_trips():
    ema = LoRAEMA(parameters(0.5), decay=0.9)
    saved = ema.state_dict()
    ema.update(parameters(9.0))
    ema.load_state_dict(saved)
    for name, tensor in ema.state_dict().items():
        assert torch.allclose(tensor, saved[name])


def test_load_state_dict_rejects_a_mismatched_adapter():
    ema = LoRAEMA(parameters(0.0), decay=0.9)
    with pytest.raises(AssertionError, match="does not match"):
        ema.load_state_dict({"unexpected": torch.zeros(1)})
