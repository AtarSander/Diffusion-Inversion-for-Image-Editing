"""Scientific checks for the standalone steering package, without pretrained weights."""

import copy
import math

import pytest
import torch

from diff_inversion.steering.directions import addition_offsets, fit_directions, select_layers
from diff_inversion.steering.hooks import REPRESENTATIONS


def records():
    result = []
    # Distinct methods expose accidental mixing of source features.
    for label, value in (
        ("positive", 3.0),
        ("positive", 5.0),
        ("negative", 1.0),
        ("negative", 1.0),
    ):
        row = {"label": label}
        for method, gain in (("DDIM", 1.0), ("MW", 2.0)):
            row[method] = {
                "features": {
                    rep: {"up_blocks.0": torch.tensor([[value * gain, 0.0], [0.0, value * gain]])}
                    for rep in REPRESENTATIONS
                },
            }
        result.append(row)
    return result


def test_difference_of_means_is_stored_before_normalization():
    bundle = fit_directions(records())
    assert torch.equal(
        bundle["directions"]["SA"]["DDIM"]["up_blocks.0"]["true"],
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    )
    assert torch.equal(
        bundle["directions"]["SA"]["MW"]["up_blocks.0"]["true"],
        torch.tensor([[6.0, 0.0], [0.0, 6.0]]),
    )


def test_gaussian_orientation_shared_and_sign_null_is_not_pair_permutation():
    bundle = fit_directions(records(), null_seed=42)
    for rep in REPRESENTATIONS:
        d, m = (bundle["directions"][rep][method]["up_blocks.0"] for method in ("DDIM", "MW"))
        assert torch.equal(d["gaussian"], m["gaussian"])
        # Pair differences are [2,4]. A balanced sign swap gives +/-1, not true=3.
        assert torch.equal(d["swap"].abs(), torch.eye(2))
        assert torch.equal(m["swap"], 2 * d["swap"])
    assert torch.equal(
        bundle["directions"]["SA"]["DDIM"]["up_blocks.0"]["gaussian"],
        fit_directions(records(), null_seed=42)["directions"]["SA"]["DDIM"]["up_blocks.0"][
            "gaussian"
        ],
    )


@pytest.mark.parametrize("control", ["true", "swap", "gaussian"])
def test_casteer_addition_uses_unit_l2_per_step_and_direct_signed_strength(control):
    bundle = fit_directions(records())
    for method, gain in (("DDIM", 1.0), ("MW", 7.0)):
        # Norms 5 and 13 distinguish L2 from RMS, squared L2 and joint time norms.
        bundle["directions"]["CA"][method]["up_blocks.0"][control] = gain * torch.tensor(
            [[3.0, 4.0, 0.0], [0.0, -5.0, 12.0]]
        )
        deltas = addition_offsets(bundle, "CA", method, control, ["up_blocks.0"], [0, 1], 2.0)
        assert torch.allclose(deltas[0]["up_blocks.0"], torch.tensor([1.2, 1.6, 0.0]))
        assert torch.allclose(deltas[1]["up_blocks.0"], torch.tensor([0.0, -10 / 13, 24 / 13]))
        negative = addition_offsets(bundle, "CA", method, control, ["up_blocks.0"], [0, 1], -2.0)
        zero = addition_offsets(bundle, "CA", method, control, ["up_blocks.0"], [0, 1], 0.0)
        for call in (0, 1):
            assert torch.equal(negative[call]["up_blocks.0"], -deltas[call]["up_blocks.0"])
            assert not zero[call]["up_blocks.0"].any()


def test_selecting_more_layers_or_calls_does_not_rescale_existing_offsets():
    bundle = fit_directions(records())
    layers = bundle["directions"]["SA"]["DDIM"]
    layers["up_blocks.1"] = copy.deepcopy(layers["up_blocks.0"])
    single = addition_offsets(bundle, "SA", "DDIM", "true", ["up_blocks.0"], [1], 2.0)
    extended = addition_offsets(bundle, "SA", "DDIM", "true", list(layers), [0, 1], 2.0)
    assert torch.equal(single[1]["up_blocks.0"], extended[1]["up_blocks.0"])


def test_near_zero_direction_uses_the_official_normalization_floor():
    bundle = fit_directions(records())
    bundle["directions"]["CA"]["DDIM"]["up_blocks.0"]["true"][0] = torch.tensor([6e-10, 8e-10])
    offsets = addition_offsets(bundle, "CA", "DDIM", "true", ["up_blocks.0"], [0], 1.0)
    assert torch.allclose(offsets[0]["up_blocks.0"], torch.tensor([0.06, 0.08]))


def test_zero_direction_is_not_replaced_by_a_unit_vector():
    source = records()
    for row in source:
        for method in ("DDIM", "MW"):
            for rep in REPRESENTATIONS:
                row[method]["features"][rep]["up_blocks.0"].zero_()
    offsets = addition_offsets(
        fit_directions(source), "CA", "DDIM", "true", ["up_blocks.0"], [0], 1.0
    )
    assert not offsets[0]["up_blocks.0"].any()


def test_bad_groups_or_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="even"):
        fit_directions(records()[:3])
    source = copy.deepcopy(records())
    source[0]["MW"]["features"]["SA"]["up_blocks.0"][0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        fit_directions(source)


def test_layer_profiles_follow_observed_order_and_preserve_up_range():
    layers = [f"down_blocks.{i}" for i in range(24)]
    layers += [f"mid_block.{i}" for i in range(10)]
    layers += [f"up_blocks.{i}" for i in range(36)]
    assert select_layers(layers, "last36") == layers[34:]
    assert select_layers(layers, "last12") == layers[58:]
    assert select_layers(layers, "all") == layers
    with pytest.raises(ValueError, match="up blocks"):
        select_layers(list(reversed(layers)), "last36")
