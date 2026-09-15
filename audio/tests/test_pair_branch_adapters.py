# ABOUTME: Checks the two-adapter mechanism the pair-branch loss depends on: switching routes the
# ABOUTME: forward, and BOTH adapters still receive gradient despite set_adapter freezing one.

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
