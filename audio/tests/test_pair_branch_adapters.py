# ABOUTME: Checks the two-adapter mechanism the pair-branch loss depends on: switching routes the
# ABOUTME: forward, and BOTH adapters still receive gradient despite set_adapter freezing one.

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model


def two_adapter_net():
    """A tiny model carrying a `cond` and an `uncond` adapter, both perturbing the output."""
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["0", "1"], init_lora_weights=True)
    inject_adapter_in_model(cfg, net, adapter_name="cond")
    inject_adapter_in_model(cfg, net, adapter_name="uncond")
    layers = [m for m in net.modules() if hasattr(m, "lora_A")]
    for m in layers:
        for name in ("cond", "uncond"):
            torch.nn.init.normal_(m.lora_B[name].weight, std=0.5)
    return net, layers


def activate(net, layers, name):
    """Route through one adapter, then restore requires_grad on every LoRA tensor.

    `set_adapter` freezes the adapters it deactivates; without this restore the unconditional
    branch would silently never train.
    """
    for m in layers:
        m.set_adapter(name)
    for pname, p in net.named_parameters():
        p.requires_grad_("lora" in pname.lower())


def test_switching_routes_the_forward():
    """Each adapter gives a different output, and switching back is exact."""
    torch.manual_seed(0)
    net, layers = two_adapter_net()
    x = torch.randn(1, 8)
    activate(net, layers, "cond")
    a = net(x)
    activate(net, layers, "uncond")
    b = net(x)
    assert not torch.allclose(a, b)
    activate(net, layers, "cond")
    assert torch.allclose(net(x), a)


def test_both_adapters_receive_gradient():
    """The pair-branch loss must train BOTH adapters, one per branch."""
    torch.manual_seed(0)
    net, layers = two_adapter_net()
    x = torch.randn(4, 8)

    activate(net, layers, "cond")
    loss = (net(x) - torch.randn(4, 8)) .pow(2).mean()
    activate(net, layers, "uncond")
    loss = loss + (net(x) - torch.randn(4, 8)).pow(2).mean()
    loss.backward()

    grads = {n: p.grad for n, p in net.named_parameters() if "lora_B" in n and p.grad is not None}
    cond = [g for n, g in grads.items() if "cond" in n and "uncond" not in n]
    uncond = [g for n, g in grads.items() if "uncond" in n]
    assert cond and uncond, f"missing a branch: {sorted(grads)}"
    assert any(g.abs().sum() > 0 for g in cond), "conditional adapter got no gradient"
    assert any(g.abs().sum() > 0 for g in uncond), "unconditional adapter got no gradient"


def test_set_adapter_alone_would_starve_one_branch():
    """Without the requires_grad restore, the inactive adapter is frozen -- the trap this guards."""
    torch.manual_seed(0)
    net, layers = two_adapter_net()
    for m in layers:
        m.set_adapter("uncond")
    frozen = [n for n, p in net.named_parameters() if "lora" in n and "cond" in n
              and "uncond" not in n and not p.requires_grad]
    assert frozen, "expected set_adapter to freeze the conditional adapter"


def test_pair_loader_routes_and_restores(tmp_path):
    """`attach_pair_inversion_lora` loads both branches, routes between them, and can be turned off.

    The conditional and unconditional sidecars must name DIFFERENT adapters; if they shared a
    name, the second load would overwrite the first and both branches would silently be the same
    weights -- which the loader asserts against.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peft import get_peft_model_state_dict

    from src.inversion_lora.apply_lora import attach_pair_inversion_lora

    torch.manual_seed(0)
    lora = {"r": 2, "lora_alpha": 2, "target_modules": ["0", "1"], "init_lora_weights": True}
    for suffix, name in (("", "inversion"), ("_uncond", "inversion_uncond")):
        donor = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        inject_adapter_in_model(LoraConfig(**lora), donor, adapter_name=name)
        for m in donor.modules():
            if hasattr(m, "lora_B"):
                torch.nn.init.normal_(m.lora_B[name].weight, std=0.7)
        torch.save(get_peft_model_state_dict(donor, adapter_name=name),
                   tmp_path / f"ckpt{suffix}.pt")
        (tmp_path / f"ckpt{suffix}.json").write_text(
            json.dumps({"adapter_name": name, "lora": lora})
        )

    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    set_enabled, select = attach_pair_inversion_lora(net, tmp_path / "ckpt.pt")
    x = torch.randn(1, 8)

    off = net(x)                      # injected disabled
    set_enabled(True)
    select("cond")
    cond = net(x)
    select("uncond")
    uncond = net(x)
    assert not torch.allclose(off, cond), "enabling the adapter changed nothing"
    assert not torch.allclose(cond, uncond), "both branches produced identical output"
    select("cond")
    assert torch.allclose(net(x), cond), "routing is not stable"
    set_enabled(False)
    assert torch.allclose(net(x), off), "disabling did not restore the base model"


def test_pair_loader_rejects_a_missing_unconditional_branch(tmp_path):
    """A single-adapter checkpoint must not be loaded as a pair, silently running one branch."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.inversion_lora.apply_lora import attach_pair_inversion_lora

    lora = {"r": 2, "lora_alpha": 2, "target_modules": ["0"], "init_lora_weights": True}
    torch.save({}, tmp_path / "solo.pt")
    (tmp_path / "solo.json").write_text(json.dumps({"adapter_name": "inversion", "lora": lora}))
    net = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(FileNotFoundError, match="_uncond"):
        attach_pair_inversion_lora(net, tmp_path / "solo.pt")
