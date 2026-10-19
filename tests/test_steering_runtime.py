"""Actual tiny transformer forwards: hook boundaries, time indexing and LoRA routing."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler
from diffusers.models.attention import BasicTransformerBlock

from diff_inversion.steering.hooks import feature_hooks
from diff_inversion.steering.runtime import SteeringRuntime, pipeline_state


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.disable_adapters = True

    def enable_adapters(self, enabled):
        self.disable_adapters = not enabled


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = torch.nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=4, num_attention_heads=1, attention_head_dim=4, cross_attention_dim=4
                )
                for _ in range(2)
            ]
        )
        self.adapter = Adapter()
        self.calls = []

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, latent, timestep, encoder_hidden_states, **kwargs):
        self.calls.append(
            (float(timestep), not self.adapter.disable_adapters, latent.detach().clone())
        )
        hidden = latent.flatten(2).transpose(1, 2)
        for block in self.up_blocks:
            hidden = block(hidden, encoder_hidden_states=encoder_hidden_states)
        result = hidden.transpose(1, 2).reshape_as(latent) * 0.01
        if not self.adapter.disable_adapters:
            result = result + 0.002
        return (result,)


def make_example():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = SimpleNamespace(unet=TinyUNet().eval(), scheduler=scheduler, device=torch.device("cpu"))
    cond = {"prompt_embeds": torch.ones(1, 2, 4), "negative_prompt_embeds": torch.zeros(1, 2, 4)}
    return pipe, cond, torch.ones(1, 4, 2, 2) * 0.1


@pytest.fixture
def example():
    return make_example()


def test_capture_is_observer_and_block_means_whole_update(example):
    pipe, cond, latent = example
    block = pipe.unet.up_blocks[0]
    inputs, outputs = [], []
    pre = block.register_forward_pre_hook(lambda _m, args: inputs.append(args[0].detach().clone()))
    post = block.register_forward_hook(lambda _m, _args, out: outputs.append(out.detach().clone()))
    capture, moments = {}, {}
    from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl

    plain = predict_noise_sdxl(pipe, latent, torch.tensor(1), cond, 1.0)
    with feature_hooks(pipe.unet, captures=capture, moments=moments):
        observed = predict_noise_sdxl(pipe, latent, torch.tensor(1), cond, 1.0)
    pre.remove()
    post.remove()
    assert torch.equal(plain, observed)
    delta = outputs[-1].float() - inputs[-1].float()
    assert torch.equal(capture["block"]["up_blocks.0"], delta[1:].mean(1))
    assert torch.equal(
        moments["block"]["up_blocks.0"], delta[1:].square().mean((1, 2), dtype=torch.float64)
    )
    assert set(capture) == {"CA", "SA", "block"}
    assert not torch.equal(capture["SA"]["up_blocks.0"], capture["block"]["up_blocks.0"])


@pytest.mark.parametrize("representation", ["CA", "SA", "block"])
def test_only_conditional_branch_changed_and_hooks_removed(example, representation):
    pipe, cond, latent = example
    from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl_branches

    plain = predict_noise_sdxl_branches(pipe, latent, torch.tensor(1), cond, 1.0)
    with feature_hooks(
        pipe.unet,
        representation=representation,
        deltas={"up_blocks.0": torch.tensor([1.0, -2.0, 3.0, 0.5])},
    ):
        changed = predict_noise_sdxl_branches(pipe, latent, torch.tensor(1), cond, 1.0)
    assert torch.equal(plain[0], changed[0])
    assert not torch.equal(plain[1], changed[1])
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in pipe.unet.modules()
    )


def test_direct_reader_uses_matching_inverse_states_and_free_reader_rolls_out(example):
    pipe, cond, latent = example
    runtime = SteeringRuntime(pipe, steps=4)
    states = runtime.invert(latent, cond, "MW")
    assert len(states) == 5
    assert all(enabled for _, enabled, _ in pipe.unet.calls)
    pipe.unet.calls.clear()
    direct = runtime.read(states, cond, reader="direct")
    for call, (_, enabled, value) in enumerate(pipe.unet.calls):
        assert not enabled
        assert torch.equal(value[:1], states[4 - call])
    assert direct["reconstruction"] is None
    pipe.unet.calls.clear()
    free = runtime.read(states, cond, reader="free")
    assert torch.equal(pipe.unet.calls[0][2][:1], states[-1])
    assert not torch.equal(pipe.unet.calls[1][2][:1], states[-2])
    assert free["features"]["SA"]["up_blocks.0"].shape == (4, 4)
    assert free["moments"]["SA"]["up_blocks.0"].shape == (4,)
    assert all(not enabled for _, enabled, _ in pipe.unet.calls)


def test_ddim_inverse_and_targets_have_lora_off_and_zero_strength_is_exact(example):
    pipe, cond, latent = example
    runtime = SteeringRuntime(pipe, steps=4)
    runtime.invert(latent, cond, "DDIM")
    assert all(not enabled for _, enabled, _ in pipe.unet.calls)
    initial = latent.clone()
    base = runtime.generate(latent, cond)
    zero = runtime.generate(
        latent,
        cond,
        representation="SA",
        offsets={call: {"up_blocks.0": torch.zeros(4)} for call in range(4)},
    )
    assert torch.equal(base, zero)
    assert torch.equal(latent, initial)
    assert pipe.unet.adapter.disable_adapters


def test_hooks_and_pipeline_state_restored_on_error(example):
    pipe, _, _ = example
    original = pipe.scheduler
    with pytest.raises(RuntimeError, match="forced"):
        with pipeline_state(pipe, lora=True), feature_hooks(pipe.unet, captures={}):
            pipe.scheduler = None
            raise RuntimeError("forced")
    assert pipe.scheduler is original
    assert pipe.unet.adapter.disable_adapters
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in pipe.unet.modules()
    )


def test_real_peft_adapter_flags_and_trainability_restored(example):
    from peft import LoraConfig, inject_adapter_in_model

    pipe, _, _ = example
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=1, target_modules=["to_q"]), pipe.unet)
    layers = [module for module in pipe.unet.modules() if hasattr(module, "enable_adapters")]
    for layer in layers:
        layer.enable_adapters(False)
    pipe.unet.requires_grad_(False)
    with pipeline_state(pipe, lora=True):
        assert all(not layer.disable_adapters for layer in layers)
    assert all(layer.disable_adapters for layer in layers)
    assert all(not parameter.requires_grad for parameter in pipe.unet.parameters())
