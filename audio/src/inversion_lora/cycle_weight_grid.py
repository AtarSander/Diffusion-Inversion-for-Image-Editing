# ABOUTME: Prints the SAO cosine-grid affine coefficients A, B so the cycle loss's implicit B^2
# ABOUTME: weight against L_inv is known before any training is launched. Scheduler only, no GPU.

import json
import subprocess
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.stable_audio import MODEL_ID, ExactDPMSolver  # noqa: E402


def main(
    model_id: str = MODEL_ID,
    num_inference_steps: int = 100,
    output_dir: str = "output/sao_cycle_weight",
) -> None:
    """Report B^2 across the Stable Audio cosine grid, the cycle loss's weight against L_inv.

    Args:
        model_id: Hub id of the pipeline (only its scheduler is loaded).
        num_inference_steps: Grid length; must match the trajectory dataset.
        output_dir: Destination, relative to `audio/`.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    from diffusers import CosineDPMSolverMultistepScheduler

    sched = CosineDPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    sched.set_timesteps(num_inference_steps)
    solver = ExactDPMSolver(sched)

    n = solver.invertible_steps
    A, B, t = [], [], []
    for i in range(n):
        a, b = solver.coefficients(i)
        A.append(float(a))
        B.append(float(b))
        t.append(solver.timesteps[i])
    A, B, t = np.array(A), np.array(B), np.array(t)
    B2, BA = B**2, np.abs(B / A)

    print(f"grid: {num_inference_steps} steps, {n} invertible "
          f"(sigma {float(solver.sigmas[0]):.1f} -> {float(solver.sigmas[-1]):.3f})")
    print(f"timesteps: {t[0]:.4f} (noisiest) -> {t[-1]:.4f} (cleanest)\n")
    print(f"{'t':>8} {'sigma':>10} {'A':>10} {'B':>11} {'B^2':>11} {'|B/A|':>9}")
    idx = list(range(0, n, max(n // 8, 1))) + [n - 1]
    for i in sorted(set(idx)):
        print(f"{t[i]:>8.4f} {float(solver.sigmas[i]):>10.3f} {A[i]:>10.5f} {B[i]:>11.5f} "
              f"{B2[i]:>11.3e} {BA[i]:>9.4f}")

    print(f"\nB^2   : min {B2.min():.3e}  max {B2.max():.3e}  mean {B2.mean():.3e}")
    print(f"|B/A| : min {BA.min():.4f}  max {BA.max():.4f}  mean {BA.mean():.4f}")
    print(f"\nIf ||e-v|| ~ ||e-u|| as on SD1.4, the cycle term's weight vs L_inv is ~B^2:")
    print(f"  mean {100 * B2.mean():.3f}%  ->  lambda_cycle for parity ~ {1 / B2.mean():.1f}")
    print("\nB^2 by band (noisiest first, each 25% of the invertible grid):")
    for k, band in enumerate(np.array_split(B2, 4)):
        print(f"  band {k}: mean B^2 = {band.mean():.3e}")

    out = AUDIO_ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cycle_weight_grid_{num_inference_steps}steps.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=AUDIO_ROOT, text=True).strip(),
            "model_id": model_id, "num_inference_steps": num_inference_steps,
            "invertible_steps": n,
            "mean_B2": float(B2.mean()), "min_B2": float(B2.min()), "max_B2": float(B2.max()),
            "lambda_parity": float(1 / B2.mean()),
            "mean_abs_B_over_A": float(BA.mean()), "max_abs_B_over_A": float(BA.max()),
            "timesteps": t.tolist(), "A": A.tolist(), "B": B.tolist(),
        }, f, indent=2)
    logger.success("Wrote {}", path)


if __name__ == "__main__":
    fire.Fire(main)
