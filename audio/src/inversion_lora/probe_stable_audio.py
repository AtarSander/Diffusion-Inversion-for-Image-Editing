# ABOUTME: Probe the Stable Audio Open teacher before porting the inversion LoRA: check which
# ABOUTME: noise schedule the DiT is actually being fed, and measure the DDIM-inversion shift gap.

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

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.stable_audio import (  # noqa: E402
    MODEL_ID,
    SCHEDULES,
    StableAudioTeacher,
    decode_to_audio,
    load_teacher,
)

NUM_QUARTILES = 4


def git_sha() -> str:
    """Return the current commit SHA so the probe's numbers are traceable to their code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
    ).strip()


@torch.no_grad()
def shift_gap(
    teacher: StableAudioTeacher,
    trajectory: torch.Tensor,
    outputs: torch.Tensor,
    timesteps: list[float],
    text_audio: torch.Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Measure the error DDIM-style inversion makes, per transition.

    Inverting the step that produced `trajectory[i + 1]` needs the teacher's prediction at the
    noisier `trajectory[i]`, which is what inversion is solving for; it substitutes the prediction
    at `trajectory[i + 1]` instead. That substitution error is what the inversion LoRA removes.

    Args:
        teacher: The loaded teacher.
        trajectory: Latents `[N + 1, C, L]`, noisiest first.
        outputs: Teacher predictions `[N, C, L]` at the noisier end of each transition.
        timesteps: The sampling grid, one entry per transition.
        text_audio: Cross-attention states `[1, S, D]`.
        batch_size: Transitions per forward pass.

    Returns:
        Per-transition `rms` error, `signed` mean error and teacher `magnitude`.
    """
    scheduler = teacher.model.scheduler
    rms, signed, magnitude = [], [], []
    for start in range(0, len(timesteps), batch_size):
        stop = min(start + batch_size, len(timesteps))
        x_clean = trajectory[start + 1 : stop + 1].to(teacher.device)
        t = torch.tensor(timesteps[start:stop], device=teacher.device)
        # The cosine scheduler preconditions the DiT's input by 1/sqrt(sigma^2 + sigma_data^2), so
        # the shifted call must be scaled by the sigma of the timestep it claims to be at, not by
        # the cleaner latent's own. DPMSolverMultistep has no preconditioning and skips this.
        if hasattr(scheduler, "precondition_inputs"):
            sigma = scheduler.sigmas[start:stop].to(teacher.device).reshape(-1, 1, 1)
            x_clean = scheduler.precondition_inputs(x_clean, sigma)
        shifted = teacher.forward(x_clean, t, text_audio).cpu()
        gap = shifted - outputs[start:stop]
        rms.append(gap.flatten(1).pow(2).mean(dim=1).sqrt().numpy())
        signed.append(gap.flatten(1).mean(dim=1).numpy())
        magnitude.append(outputs[start:stop].flatten(1).pow(2).mean(dim=1).sqrt().numpy())
    return {
        "rms": np.concatenate(rms),
        "signed": np.concatenate(signed),
        "magnitude": np.concatenate(magnitude),
    }


def quartile_table(gap: dict[str, np.ndarray], timesteps: list[float]) -> dict[str, dict]:
    """Summarise the per-transition gap by quartile of the sampling grid, noisiest first.

    The two schedules put their timesteps on different scales, so the quartiles are taken over
    step index rather than timestep value, which makes the two tables comparable.

    Args:
        gap: Output of `shift_gap`.
        timesteps: The sampling grid.

    Returns:
        One row per quartile, keyed `q1`..`q4`.
    """
    steps = len(timesteps)
    index = np.arange(steps)
    quartile = np.clip(index * NUM_QUARTILES // steps, 0, NUM_QUARTILES - 1)
    rows = {}
    for q in range(NUM_QUARTILES):
        mask = quartile == q
        e_rms = float(gap["rms"][mask].mean())
        sigma = float(gap["magnitude"][mask].mean())
        rows[f"q{q + 1}"] = {
            "t_first": timesteps[int(index[mask][0])],
            "t_last": timesteps[int(index[mask][-1])],
            "teacher_rms": sigma,
            "e_RMS": e_rms,
            "e_rel": e_rms / sigma,
            "bias_ratio": float(np.abs(gap["signed"][mask]).mean()) / e_rms,
        }
    return rows


def main(
    device: str = "cuda:0",
    schedules: tuple[str, ...] = SCHEDULES,
    num_inference_steps: int = 20,
    prompt: str = "A gentle acoustic guitar melody with soft percussion.",
    duration_s: float | None = None,
    batch_size: int = 4,
    seed: int = 42,
    output_dir: str = "output/sao_probe",
    model_id: str = MODEL_ID,
) -> None:
    """Compare the two candidate Stable Audio teachers and report their inversion shift gap.

    Writes one wav per schedule plus a JSON of the numbers, so the schedule question can be
    settled by listening as well as by the table. Defaults to a 20-step grid, which runs in about
    a minute per schedule; the editing baselines use 100.

    Args:
        device: Torch device.
        schedules: Which schedules to probe, from `SCHEDULES`.
        num_inference_steps: Length of the sampling grid.
        prompt: Caption to sample from.
        duration_s: Duration the timing conditioning declares; None uses the model's maximum.
        batch_size: Transitions per forward pass in the gap measurement.
        seed: Seed for the initial latent.
        output_dir: Directory for the wavs and the JSON, relative to `audio/`.
        model_id: Hub id of the pipeline.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    torch_device = torch.device(device)
    if torch_device.type != "cpu":
        torch.cuda.set_device(torch_device)

    out_dir = AUDIO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for schedule in schedules:
        teacher = load_teacher(
            model_id, torch_device, num_inference_steps, duration_s=duration_s, schedule=schedule
        )
        scheduler = teacher.model.scheduler
        logger.info(
            "{}: {} timesteps {:.4g}..{:.4g}, sigmas {:.4g}..{:.4g}",
            schedule,
            type(scheduler).__name__,
            float(scheduler.timesteps[0]),
            float(scheduler.timesteps[-1]),
            float(scheduler.sigmas[0]),
            float(scheduler.sigmas[-2]),
        )

        text_audio = teacher.encode_prompt(prompt)
        trajectory, outputs, timesteps = teacher.reverse_trajectory(text_audio, seed)
        logger.info(
            "{}: latents {} teacher predictions {}, teacher RMS {:.4g} at the noisy end and "
            "{:.4g} at the clean end",
            schedule,
            tuple(trajectory.shape),
            tuple(outputs.shape),
            float(outputs[0].pow(2).mean().sqrt()),
            float(outputs[-1].pow(2).mean().sqrt()),
        )

        audio = decode_to_audio(teacher, trajectory[-1:])[0]
        wav_path = out_dir / f"{schedule}_{num_inference_steps}steps.wav"
        torchaudio.save(
            str(wav_path), audio, sample_rate=teacher.pipe.vae.config.sampling_rate
        )

        gap = shift_gap(teacher, trajectory, outputs, timesteps, text_audio, batch_size)
        rows = quartile_table(gap, timesteps)
        report[schedule] = {
            "scheduler": type(scheduler).__name__,
            "timesteps": [float(t) for t in timesteps],
            "duration_s": teacher.duration_s,
            "audio_rms": float(audio.pow(2).mean().sqrt()),
            "audio_peak": float(audio.abs().max()),
            "wav": str(wav_path.relative_to(AUDIO_ROOT)),
            "quartiles": rows,
            "per_step_e_RMS": gap["rms"].tolist(),
            "per_step_teacher_rms": gap["magnitude"].tolist(),
        }

        print(f"\n{schedule}: {type(scheduler).__name__}, {len(timesteps)} steps, no CFG")
        print(
            f"  decoded audio RMS {report[schedule]['audio_rms']:.4f} "
            f"peak {report[schedule]['audio_peak']:.4f} -> {wav_path.name}"
        )
        print(
            f"  {'quartile':>10} {'t range':>18} {'teacher':>9} {'e_RMS':>9} {'e_rel':>8} "
            f"{'bias':>7}"
        )
        for key, row in rows.items():
            print(
                f"  {key:>10} {row['t_first']:>8.4g}..{row['t_last']:<8.4g} "
                f"{row['teacher_rms']:>9.4f} {row['e_RMS']:>9.4f} {row['e_rel']:>7.2%} "
                f"{row['bias_ratio']:>7.3f}"
            )

        del teacher
        torch.cuda.empty_cache()

    json_path = out_dir / f"probe_{num_inference_steps}steps.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "git_sha": git_sha(),
                "prompt": prompt,
                "seed": seed,
                "model_id": model_id,
                "schedules": report,
            },
            f,
            indent=2,
        )
    logger.success("Wrote {}", json_path)


if __name__ == "__main__":
    fire.Fire(main)
