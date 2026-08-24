# ABOUTME: Compare Stable Audio's native cosine grid against the editing code's beta grid under a
# ABOUTME: first-order sampler with an exact inverse: audio quality, invertibility, and shift gap.

import json
import subprocess
import sys
from pathlib import Path

import fire
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.stable_audio import (  # noqa: E402
    MODEL_ID,
    SCHEDULES,
    FirstOrderSolver,
    StableAudioTeacher,
    decode_to_audio,
    load_teacher,
)

NUM_QUARTILES = 4


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance."""
    return float((a - b).norm() / b.norm())


@torch.no_grad()
def sample(
    teacher: StableAudioTeacher, solver: FirstOrderSolver, text_audio: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the first-order reverse pass, keeping every latent and every data prediction.

    Args:
        teacher: The loaded teacher.
        solver: Solver over the schedule's grid.
        text_audio: Cross-attention states.
        seed: Seed for the initial latent.

    Returns:
        `(trajectory, data)` with `trajectory[i]` noisier than `trajectory[i + 1]` and `data[i]`
        the data prediction at `trajectory[i]`, both on the CPU.
    """
    shape = (1, teacher.pipe.transformer.config.in_channels, teacher.latent_length)
    x = torch.randn(
        shape, generator=torch.Generator(device=teacher.device).manual_seed(seed),
        device=teacher.device,
    ) * solver.sigmas[0]

    trajectory, data = [x.cpu()], []
    for index in tqdm(range(len(solver.timesteps)), desc="sampling", leave=False):
        t = torch.tensor([solver.timesteps[index]], device=teacher.device)
        raw = teacher.forward(solver.model_input(x, index), t, text_audio)
        prediction = solver.data_prediction(x, raw, index)
        data.append(prediction.cpu())
        x = solver.forward(x, prediction, index)
        trajectory.append(x.cpu())
    return torch.cat(trajectory), torch.cat(data)


@torch.no_grad()
def invert(
    teacher: StableAudioTeacher,
    solver: FirstOrderSolver,
    clean: torch.Tensor,
    text_audio: torch.Tensor,
    data: torch.Tensor | None = None,
    shifted_timestep: bool = True,
) -> torch.Tensor:
    """Walk the reverse grid backwards with the exact inverse step.

    Args:
        teacher: The loaded teacher.
        solver: Solver over the schedule's grid.
        clean: The clean latent to invert.
        text_audio: Cross-attention states.
        data: When given, the reverse pass's own data predictions -- the oracle. Otherwise the
            network is queried on the cleaner latent, which is the substitution inversion makes.
        shifted_timestep: Query the network at the noisier step's timestep (the shifted-denoiser
            convention the training pairs use) rather than the cleaner latent's own.

    Returns:
        The recovered noisy latent.
    """
    x = clean.to(teacher.device)
    for index in reversed(range(solver.invertible_steps)):
        if data is not None:
            prediction = data[index : index + 1].to(teacher.device)
        else:
            # The step being undone used the network at (trajectory[index], timesteps[index]); only
            # the cleaner latent is available, so either the latent alone is substituted or the
            # timestep moves with it.
            step = index if shifted_timestep else min(index + 1, len(solver.timesteps) - 1)
            t = torch.tensor([solver.timesteps[step]], device=teacher.device)
            raw = teacher.forward(solver.model_input(x, step), t, text_audio)
            prediction = solver.data_prediction(x, raw, step)
        x = solver.inverse(x, prediction, index)
    return x


@torch.no_grad()
def shift_gap(
    teacher: StableAudioTeacher,
    solver: FirstOrderSolver,
    trajectory: torch.Tensor,
    data: torch.Tensor,
    text_audio: torch.Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Per-transition error of the substitution the inversion makes, in data-prediction space."""
    steps = solver.invertible_steps
    rms, magnitude = [], []
    for start in range(0, steps, batch_size):
        stop = min(start + batch_size, steps)
        x_clean = trajectory[start + 1 : stop + 1].to(teacher.device)
        index = torch.arange(start, stop)
        scaled = torch.cat(
            [solver.model_input(x_clean[k : k + 1], int(i)) for k, i in enumerate(index)]
        )
        t = torch.tensor([solver.timesteps[int(i)] for i in index], device=teacher.device)
        raw = teacher.forward(scaled, t, text_audio)
        shifted = torch.cat(
            [
                solver.data_prediction(x_clean[k : k + 1], raw[k : k + 1], int(i))
                for k, i in enumerate(index)
            ]
        ).cpu()
        gap = shifted - data[start:stop]
        rms.append(gap.flatten(1).pow(2).mean(dim=1).sqrt().numpy())
        magnitude.append(data[start:stop].flatten(1).pow(2).mean(dim=1).sqrt().numpy())
    return {"rms": np.concatenate(rms), "magnitude": np.concatenate(magnitude)}


def quartiles(gap: dict[str, np.ndarray]) -> list[dict]:
    """Mean gap per quartile of the grid, noisiest first."""
    steps = len(gap["rms"])
    index = np.arange(steps)
    band = np.clip(index * NUM_QUARTILES // steps, 0, NUM_QUARTILES - 1)
    rows = []
    for q in range(NUM_QUARTILES):
        mask = band == q
        e, sigma = float(gap["rms"][mask].mean()), float(gap["magnitude"][mask].mean())
        rows.append({"quartile": f"q{q + 1}", "e_RMS": e, "data_RMS": sigma, "e_rel": e / sigma})
    return rows


def main(
    device: str = "cuda:0",
    num_inference_steps: int = 100,
    prompt: str = "A gentle acoustic guitar melody with soft percussion.",
    seed: int = 42,
    batch_size: int = 4,
    output_dir: str = "output/sao_schedules",
    model_id: str = MODEL_ID,
) -> None:
    """Score both schedules on invertibility, shift gap and audio, and write the comparison.

    Args:
        device: Torch device.
        num_inference_steps: Length of the grid.
        prompt: Caption to sample from.
        seed: Seed for the initial latent.
        batch_size: Transitions per forward in the gap measurement.
        output_dir: Destination for the wavs and JSON, relative to `audio/`.
        model_id: Hub id of the pipeline.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    torch_device = torch.device(device)
    if torch_device.type != "cpu":
        torch.cuda.set_device(torch_device)
    out_dir = AUDIO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for schedule in SCHEDULES:
        teacher = load_teacher(model_id, torch_device, num_inference_steps, schedule=schedule)
        solver = FirstOrderSolver(teacher.model.scheduler)
        text_audio = teacher.encode_prompt(prompt)
        logger.info(
            "{}: sigma {:.4g}..{:.4g}, t {:.4g}..{:.4g}, preconditioned={}",
            schedule, float(solver.sigmas[0]), float(solver.sigmas[-2]),
            solver.timesteps[0], solver.timesteps[-1], solver.preconditioned,
        )

        trajectory, data = sample(teacher, solver, text_audio, seed)
        audio = decode_to_audio(teacher, trajectory[-1:])[0]
        wav = out_dir / f"{schedule}_ode_{num_inference_steps}steps.wav"
        torchaudio.save(str(wav), audio, sample_rate=teacher.pipe.vae.config.sampling_rate)

        target = trajectory[:1]
        start = trajectory[solver.invertible_steps : solver.invertible_steps + 1]
        oracle = invert(teacher, solver, start, text_audio, data=data).cpu()
        shifted = invert(teacher, solver, start, text_audio, shifted_timestep=True).cpu()
        matched = invert(teacher, solver, start, text_audio, shifted_timestep=False).cpu()
        gap = quartiles(shift_gap(teacher, solver, trajectory, data, text_audio, batch_size))

        report[schedule] = {
            "sigma_max": float(solver.sigmas[0]),
            "sigma_min": float(solver.sigmas[-2]),
            "timestep_range": [solver.timesteps[0], solver.timesteps[-1]],
            "audio_rms": float(audio.pow(2).mean().sqrt()),
            "audio_peak": float(audio.abs().max()),
            "wav": str(wav.relative_to(AUDIO_ROOT)),
            "recovery": {
                "oracle": relative(oracle, target),
                "shifted_timestep": relative(shifted, target),
                "matched_timestep": relative(matched, target),
            },
            "shift_gap": gap,
        }

        print(f"\n{schedule}: sigma {float(solver.sigmas[0]):.4g}..{float(solver.sigmas[-2]):.4g}, "
              f"audio RMS {report[schedule]['audio_rms']:.4f} peak "
              f"{report[schedule]['audio_peak']:.4f} -> {wav.name}")
        print("  recovery of the initial noise (relative L2):")
        for key, value in report[schedule]["recovery"].items():
            print(f"    {key:>18}: {value:.6f}")
        print(f"  {'quartile':>10} {'data RMS':>9} {'e_RMS':>9} {'e_rel':>8}")
        for row in gap:
            print(f"  {row['quartile']:>10} {row['data_RMS']:>9.4f} {row['e_RMS']:>9.4f} "
                  f"{row['e_rel']:>7.2%}")

        del teacher
        torch.cuda.empty_cache()

    path = out_dir / f"schedules_{num_inference_steps}steps.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=AUDIO_ROOT, text=True
                ).strip(),
                "prompt": prompt,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "schedules": report,
            },
            f,
            indent=2,
        )
    logger.success("Wrote {}", path)


if __name__ == "__main__":
    fire.Fire(main)
