# ABOUTME: Unit tests for face-steering aggregation: paired deltas vs base and
# ABOUTME: pairwise NaN dropping for images where no face was detected.
import math

from diff_inversion.steering.score_faces import aggregate_faces


def row(prompt, method, strength, p_probe, id_sim=0.9):
    full = {
        "prompt": prompt,
        "seed": 42,
        "method": method,
        "strength": strength,
        "p_target": 0.5,
        "clip_t": 0.2,
        "p_probe": p_probe,
        "id_sim": 1.0 if method == "base" else id_sim,
    }
    if method != "base":
        full |= {"representation": "SA", "control": "true"}
    return full


def test_paired_deltas_and_nan_dropping():
    rows = [
        row("p1", "base", 0.0, 0.2),
        row("p2", "base", 0.0, 0.4),
        row("p1", "DDIM", 1.0, 0.7),
        row("p2", "DDIM", 1.0, float("nan")),
    ]
    summaries = {s["variant"]: s for s in aggregate_faces(rows)}
    variant = summaries["SA/DDIM/true/a=1.0"]
    assert variant["p_probe_n"] == 1
    assert math.isclose(variant["delta_p_probe_mean"], 0.5)
    assert variant["clip_t_n"] == 2
    assert math.isclose(variant["delta_clip_t_mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(variant["delta_id_sim_mean"], -0.1)
    base = summaries["base"]
    assert base["p_probe_n"] == 2
    assert math.isclose(base["id_sim_mean"], 1.0)


def test_all_nan_variant_stays_nan():
    rows = [
        row("p1", "base", 0.0, 0.2),
        row("p1", "MW", 2.0, float("nan")),
    ]
    summaries = {s["variant"]: s for s in aggregate_faces(rows)}
    variant = summaries["SA/MW/true/a=2.0"]
    assert variant["p_probe_n"] == 0
    assert math.isnan(variant["delta_p_probe_mean"])
    assert variant["clip_t_n"] == 1
