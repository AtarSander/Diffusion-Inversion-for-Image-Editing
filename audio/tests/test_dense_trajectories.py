# ABOUTME: Tests for the dense-sampling path (H2): the fine sigma grid nests the coarse one,
# ABOUTME: dense_coarse_pairs keeps the exact coarse step invariant, and the verifier enforces it.

import json
import sys
from pathlib import Path

import pytest
import torch
from diffusers import CosineDPMSolverMultistepScheduler

AUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.generate_trajectories_stable_audio import dense_coarse_pairs  # noqa: E402
from src.inversion_lora.stable_audio import ExactDPMSolver  # noqa: E402
from src.inversion_lora.verify_trajectories import check_sao_step  # noqa: E402

STRIDE = 10
FINE_STEPS = 21  # stride * (coarse - 1) + 1, nesting a 3-point coarse grid
COARSE_STEPS = 3


def solver_for(steps: int) -> ExactDPMSolver:
    scheduler = CosineDPMSolverMultistepScheduler()
    scheduler.set_timesteps(steps, device="cpu")
    return ExactDPMSolver(scheduler)


@pytest.fixture()
def solvers() -> tuple[ExactDPMSolver, ExactDPMSolver]:
    return solver_for(FINE_STEPS), solver_for(COARSE_STEPS)


def test_fine_grid_nests_coarse(solvers) -> None:
    fine, coarse = solvers
    assert torch.allclose(coarse.sigmas[:-1], fine.sigmas[:-1][::STRIDE], rtol=1e-5, atol=0)
    # A plain 10x grid (no shared endpoint arithmetic) does NOT nest: this is why the config
    # uses stride * (coarse - 1) + 1 points.
    plain = solver_for(COARSE_STEPS * STRIDE)
    assert not torch.allclose(coarse.sigmas[:-1], plain.sigmas[:-1][::STRIDE], rtol=1e-5, atol=0)


def test_dense_coarse_pairs_invariant(solvers) -> None:
    fine, coarse = solvers
    torch.manual_seed(0)
    trajectory = torch.randn(FINE_STEPS + 1, 2, 8)
    data = torch.randn(FINE_STEPS, 2, 8)

    rows, targets, states, gap_rel = dense_coarse_pairs(trajectory, data, coarse, STRIDE)

    assert rows.shape == (COARSE_STEPS, 2, 8)
    assert targets.shape == (COARSE_STEPS - 1, 2, 8)
    assert states.shape == (COARSE_STEPS, 2, 8)
    assert torch.equal(rows[0], trajectory[0])
    assert torch.equal(states, trajectory[::STRIDE][:COARSE_STEPS])
    assert torch.equal(targets, data[::STRIDE][: COARSE_STEPS - 1])

    # Each input is exactly one coarse reverse step from the fine state at its coarse point --
    # the invariant the coarse inversion relies on and the verifier recomputes.
    for k in range(COARSE_STEPS - 1):
        stepped = coarse.forward(states[k : k + 1], targets[k : k + 1], k)
        assert torch.allclose(stepped, rows[k + 1 : k + 2], rtol=1e-6, atol=1e-6)

    # With arbitrary data the formed input differs from the fine state at k+1: the gap is the
    # dense-vs-coarse signal and must be measured, not zero.
    assert gap_rel.shape == (COARSE_STEPS - 1,)
    assert (gap_rel > 0).all()


def test_check_sao_step_accepts_dense_and_rejects_corruption(tmp_path: Path, solvers) -> None:
    fine, coarse = solvers
    torch.manual_seed(1)
    trajectory = torch.randn(FINE_STEPS + 1, 2, 8)
    data = torch.randn(FINE_STEPS, 2, 8)
    rows, targets, states, _ = dense_coarse_pairs(trajectory, data, coarse, STRIDE)

    sample = tmp_path / "sample_000000"
    (sample / "latents").mkdir(parents=True)
    (sample / "targets").mkdir()
    torch.save(rows, sample / "latents/trajectory.pt")
    torch.save(states, sample / "latents/states.pt")
    torch.save(targets, sample / "targets/target_eps.pt")
    meta = {
        "model_id": "stabilityai/stable-audio-open-1.0",
        "num_inference_steps": COARSE_STEPS,
        "input_space": "exact_coarse_step",
    }
    timesteps = list(coarse.timesteps)[1:]
    cache = {(meta["model_id"], COARSE_STEPS): coarse}

    assert check_sao_step(sample, meta, rows, targets, timesteps, cache) == []

    corrupted = rows.clone()
    corrupted[1] += 1.0
    problems = check_sao_step(sample, meta, corrupted, targets, timesteps, cache)
    assert problems and "invariant" in problems[0]

    # A plain (non-dense) sample checks against its own previous rows.
    plain_meta = {k: v for k, v in meta.items() if k != "input_space"}
    plain_rows = [rows[:1]]
    for k in range(COARSE_STEPS - 1):
        plain_rows.append(coarse.forward(plain_rows[-1], targets[k : k + 1], k))
    plain_rows = torch.cat(plain_rows)
    assert check_sao_step(sample, plain_meta, plain_rows, targets, timesteps, cache) == []
