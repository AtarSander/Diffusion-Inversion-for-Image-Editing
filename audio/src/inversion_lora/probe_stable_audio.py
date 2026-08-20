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
from diffusers import (
    CosineDPMSolverMultistepScheduler,
    DPMSolverMultistepScheduler,
    StableAudioPipeline,
)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

MODEL_ID = "stabilityai/stable-audio-open-1.0"
NUM_QUARTILES = 4

# The editing code's `ddim` mode replaces Stable Audio's native cosine scheduler with
# DPMSolverMultistepScheduler.from_config(...), which drops sigma_min/sigma_max/sigma_schedule and
# falls back to a linear-beta grid. "cosine" is the schedule the model was trained on; "beta" is
# what the DDIM inversion baselines actually ran.
SCHEDULES = ("cosine", "beta")


def git_sha() -> str:
    """Return the current commit SHA so the probe's numbers are traceable to their code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
    ).strip()


def load_pipeline(
    model_id: str, device: torch.device, schedule: str, num_inference_steps: int
) -> StableAudioPipeline:
    """Load Stable Audio Open with either its native or the editing code's DDIM scheduler.

    Args:
        model_id: Hub id of the pipeline.
        device: Device to place the pipeline on.
        schedule: `cosine` for the native scheduler, `beta` for the DDIM path's replacement.
        num_inference_steps: Length of the sampling grid.

    Returns:
        The pipeline, in eval mode with its timesteps already set.
    """
    assert schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}, got {schedule!r}"
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    if schedule == "cosine":
        assert isinstance(pipe.scheduler, CosineDPMSolverMultistepScheduler), type(pipe.scheduler)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    return pipe


def conditioning(pipe: StableAudioPipeline, prompt: str, duration_s: float) -> dict:
    """Build the DiT's conditioning tensors for one prompt, exactly as the pipeline does.

    Args:
        pipe: The loaded pipeline.
        prompt: Caption to condition on.
        duration_s: Audio duration the timing conditioning declares.

    Returns:
        Dict with `text_audio` (cross-attention states), `global_states` (timing states prepended
        to the sequence) and `rotary` (the positional embedding pair).
    """
    device = pipe._execution_device
    prompt_embeds = pipe.encode_prompt([prompt], device, False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, duration_s, device, False, 1)
    global_states = torch.cat([seconds_start, seconds_end], dim=2)
    rotary = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim,
        int(pipe.transformer.config.sample_size) + global_states.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )
    return {
        "text_audio": torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1),
        "global_states": global_states,
        "rotary": rotary,
    }


def dit_forward(
    pipe: StableAudioPipeline, x: torch.Tensor, t: torch.Tensor, cond: dict
) -> torch.Tensor:
    """Run the DiT on a batch of latents with per-example timesteps.

    The pipeline passes `t.unsqueeze(0)` for a single scalar timestep; training needs one timestep
    per batch element, which the DiT supports directly since the time embedding is broadcast over
    the sequence.

    Args:
        pipe: The loaded pipeline.
        x: Latents `[B, C, L]`.
        t: Timesteps `[B]`, in the scheduler's own units.
        cond: Output of `conditioning`, built for a single prompt.

    Returns:
        The DiT's prediction, same shape as `x`.
    """
    assert x.ndim == 3, x.shape
    assert t.ndim == 1 and t.shape[0] == x.shape[0], (t.shape, x.shape)
    batch = x.shape[0]
    out = pipe.transformer(
        x,
        t,
        encoder_hidden_states=cond["text_audio"].expand(batch, -1, -1),
        global_hidden_states=cond["global_states"].expand(batch, -1, -1),
        rotary_embedding=cond["rotary"],
        return_dict=False,
    )[0]
    assert out.shape == x.shape, (out.shape, x.shape)
    return out


@torch.no_grad()
def reverse_trajectory(
    pipe: StableAudioPipeline, cond: dict, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Sample without guidance, keeping every latent and every teacher prediction.

    Args:
        pipe: The loaded pipeline, timesteps already set.
        cond: Output of `conditioning`.
        seed: Seed for the initial latent.

    Returns:
        `(latents, outputs, timesteps)` where `latents[i]` is noisier than `latents[i + 1]` and
        `outputs[i]` is the teacher's prediction at `(latents[i], timesteps[i])`.
    """
    device = pipe._execution_device
    shape = (1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size))
    latents = torch.randn(
        shape, generator=torch.Generator(device=device).manual_seed(seed), device=device
    ) * pipe.scheduler.init_noise_sigma

    trajectory = [latents.cpu()]
    outputs = []
    timesteps = []
    for t in pipe.scheduler.timesteps:
        model_input = pipe.scheduler.scale_model_input(latents, t)
        prediction = dit_forward(pipe, model_input, t.reshape(1).to(device), cond)
        outputs.append(prediction.cpu())
        latents = pipe.scheduler.step(prediction, t, latents).prev_sample
        trajectory.append(latents.cpu())
        timesteps.append(float(t))

    trajectory = torch.cat(trajectory)
    outputs = torch.cat(outputs)
    assert trajectory.shape[0] == outputs.shape[0] + 1, (trajectory.shape, outputs.shape)
    return trajectory, outputs, timesteps


@torch.no_grad()
def shift_gap(
    pipe: StableAudioPipeline,
    trajectory: torch.Tensor,
    outputs: torch.Tensor,
    timesteps: list[float],
    cond: dict,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Measure the error DDIM inversion makes, per transition.

    Inverting the step that produced `trajectory[i + 1]` needs the teacher's prediction at the
    noisier `trajectory[i]`, which is what inversion is solving for; it substitutes the prediction
    at `trajectory[i + 1]` instead. That substitution error is what the inversion LoRA removes.

    Args:
        pipe: The loaded pipeline.
        trajectory: Latents `[N + 1, C, L]`, noisiest first.
        outputs: Teacher predictions `[N, C, L]` at the noisier end of each transition.
        timesteps: The sampling grid, one entry per transition.
        cond: Output of `conditioning`.
        batch_size: Transitions per forward pass.

    Returns:
        Per-transition `rms` error, `signed` mean error and teacher `magnitude`.
    """
    device = pipe._execution_device
    rms, signed, magnitude = [], [], []
    for start in range(0, len(timesteps), batch_size):
        stop = min(start + batch_size, len(timesteps))
        x_clean = trajectory[start + 1 : stop + 1].to(device)
        t = torch.tensor(timesteps[start:stop], device=device)
        # The cosine scheduler preconditions the DiT's input by 1/sqrt(sigma^2 + sigma_data^2),
        # so the shifted call must be scaled by the sigma of the timestep it claims to be at, not
        # by the cleaner latent's own. DPMSolverMultistep has no preconditioning and skips this.
        if hasattr(pipe.scheduler, "precondition_inputs"):
            sigma = pipe.scheduler.sigmas[start:stop].to(device).reshape(-1, 1, 1)
            x_clean = pipe.scheduler.precondition_inputs(x_clean, sigma)
        shifted = dit_forward(pipe, x_clean, t, cond).cpu()
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
    duration_s: float = 47.0,
    batch_size: int = 4,
    seed: int = 42,
    output_dir: str = "output/sao_probe",
    model_id: str = MODEL_ID,
) -> None:
    """Compare the two candidate Stable Audio teachers and report their inversion shift gap.

    Writes one wav per schedule plus a JSON of the numbers, so the schedule question can be
    settled by listening as well as by the table. Defaults to a 20-step grid, which runs in
    about a minute per schedule; the editing baselines use 100.

    Args:
        device: Torch device.
        schedules: Which schedules to probe, from `SCHEDULES`.
        num_inference_steps: Length of the sampling grid.
        prompt: Caption to sample from.
        duration_s: Duration the timing conditioning declares.
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
        pipe = load_pipeline(model_id, torch_device, schedule, num_inference_steps)
        grid = pipe.scheduler.timesteps
        logger.info(
            "{}: {} timesteps {:.4g}..{:.4g}, sigmas {:.4g}..{:.4g}",
            schedule,
            type(pipe.scheduler).__name__,
            float(grid[0]),
            float(grid[-1]),
            float(pipe.scheduler.sigmas[0]),
            float(pipe.scheduler.sigmas[-2]),
        )

        cond = conditioning(pipe, prompt, duration_s)
        trajectory, outputs, timesteps = reverse_trajectory(pipe, cond, seed)
        logger.info(
            "{}: latents {} teacher predictions {}, teacher RMS {:.4g} at the noisy end and "
            "{:.4g} at the clean end",
            schedule,
            tuple(trajectory.shape),
            tuple(outputs.shape),
            float(outputs[0].pow(2).mean().sqrt()),
            float(outputs[-1].pow(2).mean().sqrt()),
        )

        with torch.no_grad():
            audio = pipe.vae.decode(trajectory[-1:].to(torch_device)).sample
        audio = audio[:, :, : int(duration_s * pipe.vae.config.sampling_rate)][0].cpu().float()
        wav_path = out_dir / f"{schedule}_{num_inference_steps}steps.wav"
        torchaudio.save(str(wav_path), audio, sample_rate=pipe.vae.config.sampling_rate)

        gap = shift_gap(pipe, trajectory, outputs, timesteps, cond, batch_size)
        rows = quartile_table(gap, timesteps)
        report[schedule] = {
            "scheduler": type(pipe.scheduler).__name__,
            "timesteps": [float(t) for t in timesteps],
            "audio_rms": float(audio.pow(2).mean().sqrt()),
            "audio_peak": float(audio.abs().max()),
            "wav": str(wav_path.relative_to(AUDIO_ROOT)),
            "quartiles": rows,
            "per_step_e_RMS": gap["rms"].tolist(),
            "per_step_teacher_rms": gap["magnitude"].tolist(),
        }

        print(f"\n{schedule}: {type(pipe.scheduler).__name__}, {len(timesteps)} steps, no CFG")
        print(f"  decoded audio RMS {report[schedule]['audio_rms']:.4f} "
              f"peak {report[schedule]['audio_peak']:.4f} -> {wav_path.name}")
        print(f"  {'quartile':>10} {'t range':>18} {'teacher':>9} {'e_RMS':>9} {'e_rel':>8} {'bias':>7}")
        for key, row in rows.items():
            print(
                f"  {key:>10} {row['t_first']:>8.4g}..{row['t_last']:<8.4g} "
                f"{row['teacher_rms']:>9.4f} {row['e_RMS']:>9.4f} {row['e_rel']:>7.2%} "
                f"{row['bias_ratio']:>7.3f}"
            )

        del pipe
        torch.cuda.empty_cache()

    json_path = out_dir / f"probe_{num_inference_steps}steps.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"git_sha": git_sha(), "prompt": prompt, "duration_s": duration_s,
             "seed": seed, "model_id": model_id, "schedules": report},
            f,
            indent=2,
        )
    logger.success("Wrote {}", json_path)


if __name__ == "__main__":
    fire.Fire(main)
