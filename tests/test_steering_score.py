# ABOUTME: Unit tests for the steering score aggregation: paired deltas against the
# ABOUTME: base image of each (prompt, seed) and exact means on known inputs.
"""Tests for diff_inversion.steering.score.aggregate."""

import pytest

from diff_inversion.steering.score import aggregate, style_prompt, variant_key


def row(method, prompt="p", seed=1, p_target=0.5, clip_t=0.2, **variant):
    return {
        "method": method,
        "prompt": prompt,
        "seed": seed,
        "p_target": p_target,
        "clip_t": clip_t,
        **variant,
    }


def variant(**kwargs):
    defaults = {"representation": "SA", "control": "true", "strength": 2.0}
    return row(kwargs.pop("method", "DDIM"), **{**defaults, **kwargs})


def test_paired_delta_uses_matching_base():
    rows = [
        row("base", prompt="a", p_target=0.1, clip_t=0.30),
        row("base", prompt="b", p_target=0.3, clip_t=0.10),
        variant(prompt="a", p_target=0.6, clip_t=0.25),
        variant(prompt="b", p_target=0.4, clip_t=0.05),
    ]
    summary = {s["variant"]: s for s in aggregate(rows)}["SA/DDIM/true/a=2.0"]
    assert summary["n"] == 2
    assert summary["p_target_mean"] == pytest.approx(0.5)
    assert summary["delta_p_target_mean"] == pytest.approx((0.5 + 0.1) / 2)
    assert summary["delta_clip_t_mean"] == pytest.approx(-0.05)


def test_base_summary_has_zero_delta():
    rows = [row("base", p_target=0.2), variant(p_target=0.9)]
    summary = {s["variant"]: s for s in aggregate(rows)}["base"]
    assert summary["delta_p_target_mean"] == 0.0


def test_duplicate_base_rejected():
    with pytest.raises(ValueError):
        aggregate([row("base"), row("base")])


def test_sorted_by_delta_descending():
    rows = [
        row("base", p_target=0.2),
        variant(p_target=0.3, strength=1.0),
        variant(p_target=0.8, strength=4.0),
    ]
    deltas = [s["delta_p_target_mean"] for s in aggregate(rows)]
    assert deltas == sorted(deltas, reverse=True)


def test_probe_metric_aggregates_when_present():
    rows = [
        {**row("base", p_target=0.2), "p_probe": 0.1},
        {**variant(p_target=0.5), "p_probe": 0.7},
    ]
    summary = {s["variant"]: s for s in aggregate(rows)}["SA/DDIM/true/a=2.0"]
    assert summary["p_probe_mean"] == pytest.approx(0.7)
    assert summary["delta_p_probe_mean"] == pytest.approx(0.6)


def test_variant_key_and_prompt_names():
    assert variant_key(row("base")) == "base"
    assert variant_key(variant(strength=8.0)) == "SA/DDIM/true/a=8.0"
    assert style_prompt("Ukiyo_e") == "a painting in the style of ukiyo-e"
    assert style_prompt("Pop_Art") == "a painting in the style of pop art"
