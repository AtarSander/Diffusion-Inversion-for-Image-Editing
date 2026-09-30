# ABOUTME: Tests the ODE-vs-forward-diffusion statistics on a Gaussian toy whose probability-flow
# ABOUTME: ODE is exact: marginals must match while the ODE's effective noise is fully coupled.

import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inversion_lora.analyze_ode_vs_forward import (  # noqa: E402
    aggregate,
    level_metrics,
    mmd_level_indices,
    mmd_test,
    noise_stats,
)

SHAPE = (4, 16)  # one state [C, L], time along dim 1
C_DATA = 0.8  # data std of the toy x0 ~ N(0, C_DATA^2 I)


def gaussian_flow(num_samples: int, abar: torch.Tensor, seed: int = 0):
    """Exact probability-flow ODE for Gaussian data: x_t = sqrt(a^2 c^2 + s^2) u, x0 = c u.

    For x0 ~ N(0, c^2 I) the flow keeps the standardized coordinate u fixed, so every ODE state
    has exactly the forward marginal N(0, (a^2 c^2 + s^2) I) -- but its effective noise is
    proportional to x0 rather than independent of it.
    """
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn((num_samples, *SHAPE), generator=gen, dtype=torch.float64)
    a, s = abar.sqrt(), (1 - abar).sqrt()
    r = (a**2 * C_DATA**2 + s**2).sqrt()
    states = r.reshape(1, -1, 1, 1) * u.unsqueeze(1)  # [n, levels, C, L]
    return u * C_DATA, states, a, s, r


def test_forward_draws_are_the_null():
    st = noise_stats(torch.randn(64, 1024, generator=torch.Generator().manual_seed(1)), 1, 0)
    assert abs(st["std"] - 1) < 0.01
    assert abs(st["kurt"]) < 0.05
    assert abs(st["lag_t"]) < 0.01 and abs(st["lag_f"]) < 0.01


def test_noise_stats_see_structure():
    smooth = torch.randn(64, 1025, generator=torch.Generator().manual_seed(2))
    smooth = (smooth[:, 1:] + smooth[:, :-1]) / math.sqrt(2)  # lag-1 correlation 1/2 along time
    assert abs(noise_stats(smooth, 1, None)["lag_t"] - 0.5) < 0.02
    assert math.isnan(noise_stats(smooth, 1, None)["lag_f"])


def test_gaussian_flow_marginals_match_but_noise_is_coupled():
    abar = torch.linspace(0.05, 0.95, 6, dtype=torch.float64)
    x0, states, a, s, r = gaussian_flow(300, abar)
    gen = torch.Generator().manual_seed(3)
    level = 3
    z_hat = states[:, 0] / s[0]
    rows, feats_ode, feats_fwd = [], [], []
    for j in range(x0.shape[0]):
        y = a[level] * x0[j] + s[level] * torch.randn(SHAPE, generator=gen, dtype=torch.float64)
        rows.append(level_metrics(states[j, level], y, x0[j], z_hat[j], float(a[level]),
                                  float(s[level]), time_dim=1, freq_dim=0))
        feats_ode.append(states[j, level].flatten())
        feats_fwd.append(y.flatten())
    df = pd.DataFrame(rows)
    expected_std = float((r[level] - a[level] * C_DATA) / s[level])
    # Coupled: the ODE's effective noise points along x0, with the flow's shrunken magnitude.
    assert df.cosx0_ode.min() > 0.999
    assert abs(df.nstd_ode.mean() - expected_std) < 0.05
    # The forward draws carry no x0 direction and unit noise.
    assert abs(df.cosx0_fwd.mean()) < 0.02
    assert abs(df.nstd_fwd.mean() - 1) < 0.02
    # ...yet the two sets of states share one marginal, which the MMD test must not reject,
    fx, fy = torch.stack(feats_ode), torch.stack(feats_fwd)
    mmd_same, p_same, _ = mmd_test(fx, fy, num_permutations=100)
    assert p_same > 0.05
    # while a 20% scale error in the ODE states is caught.
    mmd_scaled, p_scaled, _ = mmd_test(1.2 * fx, fy, num_permutations=100)
    assert p_scaled < 0.02 and mmd_scaled > 5 * max(mmd_same, 1e-12)


def test_mmd_levels_always_include_the_last():
    assert mmd_level_indices(200, 5)[-1] == 199
    assert mmd_level_indices(11, 5) == [0, 5, 10]


def test_aggregate_merges_shards(tmp_path):
    levels, n_per, dim = 6, 10, 8
    meta = {"model": "audioldm2", "model_id": "toy", "steps": levels, "levels": list(range(levels)),
            "a": [0.5] * levels, "s": [0.8] * levels, "mmd_levels": [0, 5], "state_shape": [4, 2],
            "seed_base": 0, "noise_seed_offset": 0, "proj_seed": 0, "proj_dim": dim,
            "gap_stride": 2, "gap_every": 2, "start_index": 0, "num_samples": 2 * n_per,
            "num_shards": 2, "git_sha": "unknown"}
    out = tmp_path / "audioldm2"
    out.mkdir()
    names = ["rms_ode", "rms_fwd", "nstd_ode", "nstd_fwd", "nkurt_ode", "nkurt_fwd", "nlagt_ode",
             "nlagt_fwd", "nlagf_ode", "nlagf_fwd", "cosx0_ode", "cosx0_fwd", "cosz", "straight"]
    for shard in range(2):
        gen = torch.Generator().manual_seed(shard)
        gap = torch.rand(n_per, levels, generator=gen, dtype=torch.float64)
        gap_fwd = 10 * gap.clone()
        gap_fwd[:, 1::2] = float("nan")
        torch.save({
            "meta": {**meta, "shard": shard},
            "sample_idx": list(range(shard, 2 * n_per, 2)),
            "metrics": {k: torch.rand(n_per, levels, generator=gen, dtype=torch.float64)
                        for k in names},
            "gap_ode": gap,
            "gap_fwd": gap_fwd,
            "feats_ode": torch.randn(n_per, 2, dim, generator=gen),
            "feats_fwd": torch.randn(n_per, 2, dim, generator=gen),
        }, out / f"shard_{shard:03d}_of_002.pt")
    aggregate("audioldm2", out_root=str(tmp_path), num_permutations=10, report_levels=4)
    df = pd.read_csv(out / "levels.csv")
    assert len(df) == levels
    assert {"gap_ratio", "mmd2", "mmd_p", "rms_ode_mean", "cosz_mean"} <= set(df.columns)
    # Gap levels are the even ones, where the forward gap is 10x the ODE one by construction.
    assert abs(df.gap_ratio[0::2] - 10).max() < 1e-9
    assert df.gap_ratio[1::2].isna().all()
    assert df.mmd2.notna().sum() == 2
    assert "One-step gap" in (out / "REPORT.md").read_text()
