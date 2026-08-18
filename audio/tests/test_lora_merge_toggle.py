# ABOUTME: Pin the PEFT behaviour attach_inversion_lora's toggle depends on: merged weights must
# ABOUTME: equal the unfused branch, and disabling must restore the base layer exactly.

import sys
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Tiny(torch.nn.Module):
    """Two projections, so an adapter targeting one leaves the other alone."""

    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(16, 16)
        self.to_k = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.to_k(self.to_q(x))


def build():
    torch.manual_seed(0)
    model = Tiny()
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=2, target_modules=["to_q"]), model, adapter_name="inversion"
    )
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.1)
    return model, [m for m in model.modules() if m is not model and hasattr(m, "merge")]


def test_merged_output_matches_the_unfused_branch():
    """The whole point: folding B@A into the weight must not change what the model computes."""
    model, layers = build()
    x = torch.randn(4, 16)
    for layer in layers:
        layer.enable_adapters(True)
    unfused = model(x)
    for layer in layers:
        layer.merge(adapter_names=["inversion"])
    merged = model(x)
    assert torch.allclose(unfused, merged, atol=1e-6), (unfused - merged).abs().max()


def test_disabling_auto_unmerges_which_is_why_the_toggle_keeps_adapters_enabled():
    """PEFT's forward unmerges when disable_adapters is set, so the two must not be combined."""
    model, layers = build()
    x = torch.randn(4, 16)
    for layer in layers:
        layer.enable_adapters(True)
        layer.merge(adapter_names=["inversion"])
    assert all(layer.merged for layer in layers)

    for layer in layers:
        layer.enable_adapters(False)
    model(x)
    assert not any(layer.merged for layer in layers), (
        "PEFT no longer auto-unmerges on disable; apply_lora's ordering comment is now wrong"
    )


def test_toggling_off_restores_the_base_model_exactly():
    model, layers = build()
    x = torch.randn(4, 16)
    for layer in layers:
        layer.enable_adapters(False)
    base = model(x)

    for layer in layers:
        layer.enable_adapters(True)
        layer.merge(adapter_names=["inversion"])
    assert not torch.allclose(base, model(x), atol=1e-5), "adapter had no effect; test is vacuous"

    for layer in layers:
        layer.unmerge()
        layer.enable_adapters(False)
    assert torch.allclose(base, model(x), atol=1e-6)


def test_merge_unmerge_cycles_do_not_drift():
    """One cycle per edit, ~115 edits per run, so accumulated error has to stay negligible."""
    model, layers = build()
    x = torch.randn(4, 16)
    for layer in layers:
        layer.enable_adapters(False)
    base = model(x)
    for _ in range(200):
        for layer in layers:
            layer.enable_adapters(True)
            layer.merge(adapter_names=["inversion"])
        for layer in layers:
            layer.unmerge()
            layer.enable_adapters(False)
    assert torch.allclose(base, model(x), atol=1e-6), (base - model(x)).abs().max()
