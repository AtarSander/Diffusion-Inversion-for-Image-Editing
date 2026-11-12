"""No dataset or pretrained downloads: validate the real scheduler/adapter hook."""
from pathlib import Path

import pytest
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler
from diffusers.models import UNet2DConditionModel
from peft import LoraConfig

from diff_inversion.eval.ovg_bridge import load_upstream, make_inversion_method


@pytest.fixture(scope="module", autouse=True)
def upstream():
    load_upstream(Path(__file__).resolve().parents[1] / "external/ovg")


def model():
    torch.manual_seed(3)
    net = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), block_out_channels=(8,),
        layers_per_block=1, norm_num_groups=4, cross_attention_dim=8,
        attention_head_dim=2,
    )
    net.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]),
                    adapter_name="inversion")
    for name, parameter in net.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    net.eval().requires_grad_(False)
    net.disable_adapters()
    return net


def test_adapter_active_only_during_inversion_and_changes_output():
    net = model()
    schedule = DDIMScheduler(clip_sample=False, set_alpha_to_one=False)
    image, condition = torch.randn(1, 4, 8, 8), torch.randn(1, 2, 8)
    baseline = make_inversion_method(False)
    adapted = make_inversion_method(True)
    ref = baseline.run(image, net, schedule, condition, num_steps=3)
    out = adapted.run(image, net, schedule, condition, num_steps=3)
    assert not torch.allclose(ref["inverted_latents"], out["inverted_latents"])
    assert all(module.disable_adapters for module in net.modules()
               if hasattr(module, "lora_A"))
    expected = baseline.sample(out["inverted_latents"], net, schedule, condition, num_steps=3)
    torch.testing.assert_close(out["generated"], expected)


def test_matches_existing_inverse_scheduler():
    net = model()
    schedule = DDIMScheduler(clip_sample=False, steps_offset=1, set_alpha_to_one=False)
    image, cond = torch.randn(1, 4, 8, 8), torch.randn(1, 2, 8)
    result = make_inversion_method(False).invert(image, net, schedule, cond, num_steps=3)
    inverse = DDIMInverseScheduler.from_config(schedule.config)
    inverse.set_timesteps(3)
    expected = image
    with torch.no_grad():
        for timestep in inverse.timesteps:
            prediction = net(expected, timestep, encoder_hidden_states=cond).sample
            expected = inverse.step(prediction, timestep, expected).prev_sample
    torch.testing.assert_close(expected, result["latents"])


def test_adapter_disabled_after_exception():
    net = model()
    def fail(*args, **kwargs):
        raise RuntimeError("intentional forward failure")
    net.forward = fail
    with pytest.raises(RuntimeError, match="intentional"):
        make_inversion_method(True).invert(
            torch.randn(1, 4, 8, 8), net, DDIMScheduler(), torch.randn(1, 2, 8), num_steps=3
        )
    assert all(module.disable_adapters for module in net.modules()
               if hasattr(module, "lora_A"))


def test_upstream_nearest_degradation():
    from ovg.data.transforms import DegradationTransform, UpscaleTransform
    source = torch.arange(3 * 256 * 256).reshape(3, 256, 256).float()
    low = DegradationTransform(hr_size=256, scale_factor=16, use_blur=False,
                               noise_std=0, interpolation_mode="nearest")(source)
    torch.testing.assert_close(low, source[:, ::16, ::16])
    restored = UpscaleTransform(256, "nearest")(low)
    torch.testing.assert_close(restored, low.repeat_interleave(16, 1).repeat_interleave(16, 2))


def test_runner_import():
    from diff_inversion.eval.bbbc021 import runner_class
    assert runner_class().__name__ == "LoRABenchmarkRunner"


def test_edges2shoes_pair_order_and_no_augmentation(tmp_path):
    from PIL import Image
    from ovg.data.datasets import Edges2ShoesDataset
    image = Image.new("RGB", (512, 256), "black")
    image.paste(Image.new("RGB", (256, 256), "white"), (256, 0))
    image.save(tmp_path / "pair.png")
    manifest = tmp_path / "val.csv"
    manifest.write_text("image_path,image_id,class_idx\npair.png,pair,0\n")
    ds = Edges2ShoesDataset(manifest, root_dir=tmp_path, augmentation=False, paired_mode=True)
    first, second = ds[0], ds[0]
    torch.testing.assert_close(first["lr"], torch.full((3, 256, 256), -1.0))
    torch.testing.assert_close(first["hr"], torch.ones(3, 256, 256))
    torch.testing.assert_close(first["lr"], second["lr"])
