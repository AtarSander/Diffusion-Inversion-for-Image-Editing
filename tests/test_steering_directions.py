"""Scientific checks for the standalone steering package, without pretrained weights."""

import copy
import math

import pytest
import torch

from diff_inversion.steering.directions import fit_directions, scaled_offsets, select_layers
from diff_inversion.steering.hooks import REPRESENTATIONS


def records():
    result = []
    # Distinct methods and unequal direction norms expose incorrect shared scaling.
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
                "moments": {
                    rep: {"up_blocks.0": torch.tensor([4.0 * gain, 16.0 * gain])}
                    for rep in REPRESENTATIONS
                },
            }
        result.append(row)
    return result


def test_difference_of_means_and_full_map_reference():
    bundle = fit_directions(records())
    assert torch.equal(
        bundle["directions"]["SA"]["DDIM"]["up_blocks.0"]["true"],
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    )
    assert torch.equal(
        bundle["directions"]["SA"]["MW"]["up_blocks.0"]["true"],
        torch.tensor([[6.0, 0.0], [0.0, 6.0]]),
    )
    assert torch.allclose(
        bundle["reference_rms"]["SA"]["up_blocks.0"], torch.tensor([6.0, 24.0]).sqrt()
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


def test_common_requested_relative_dose_and_signed_strength():
    bundle = fit_directions(records())
    for method in ("DDIM", "MW"):
        deltas = scaled_offsets(bundle, "SA", method, "true", ["up_blocks.0"], [0, 1], 2.0)
        relative_energy = sum(
            float(value["up_blocks.0"].double().square().mean())
            / float(bundle["reference_rms"]["SA"]["up_blocks.0"][call]) ** 2
            for call, value in deltas.items()
        )
        assert relative_energy == pytest.approx(10 * 2.0**2, rel=1e-6)
        negative = scaled_offsets(bundle, "SA", method, "true", ["up_blocks.0"], [0, 1], -2.0)
        assert torch.equal(negative[0]["up_blocks.0"], -deltas[0]["up_blocks.0"])


def test_zero_direction_is_not_replaced_by_a_unit_vector():
    source = records()
    for row in source:
        for method in ("DDIM", "MW"):
            for rep in REPRESENTATIONS:
                row[method]["features"][rep]["up_blocks.0"].zero_()
    offsets = scaled_offsets(
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
