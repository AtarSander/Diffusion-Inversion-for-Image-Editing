# ABOUTME: Check that the shifted-denoiser training pair matches what Stable Audio's inverse solver
# ABOUTME: actually substitutes, and measure the round-trip error an exact-output oracle can reach.

import sys
from pathlib import Path

import fire
import torch
from diffusers import DPMSolverMultistepInverseScheduler
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.stable_audio import MODEL_ID, load_teacher  # noqa: E402


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance between two tensors."""
    return float((a - b).norm() / b.norm())


@torch.no_grad()
def invert(teacher, inverse, clean: torch.Tensor, text_audio: torch.Tensor, oracle=None,
           timestep_source=None) -> tuple[torch.Tensor, list[dict]]:
    """Run the inverse solver from a clean latent, recording what it feeds the model each step.

    Args:
        teacher: The loaded `StableAudioTeacher`.
        inverse: An inverse scheduler, re-gridded by the caller.
        clean: The clean latent to invert, `[1, C, L]`.
        text_audio: Cross-attention states.
        oracle: Optional list of the reverse pass's model outputs, noisiest first. When given, step
            `j` is fed `oracle[len(oracle) - 1 - j]` instead of a model call, which is exactly the
            output the corresponding reverse step used.
        timestep_source: Optional list of timesteps to pass to the model instead of the
            scheduler's own, so the training convention can be tested against the solver's.

    Returns:
        `(noisy_latent, trace)` where trace records the timestep and latent per step.
    """
    latents = clean.clone()
    trace = []
    for index, tau in enumerate(inverse.timesteps):
        if oracle is not None:
            prediction = oracle[len(oracle) - 1 - index].to(teacher.device)
        else:
            model_t = tau if timestep_source is None else timestep_source[index]
            prediction = teacher.forward(
                latents, torch.tensor([float(model_t)], device=teacher.device), text_audio
            )
        trace.append({"step": index, "tau": float(tau), "latent": latents.detach().cpu()})
        latents = inverse.step(prediction, tau, latents).prev_sample
    return latents, trace


def main(
    device: str = "cuda:0",
    num_inference_steps: int = 20,
    prompt: str = "A gentle acoustic guitar melody with soft percussion.",
    seed: int = 42,
    model_id: str = MODEL_ID,
) -> None:
    """Compare the training pairing with the inverse solver's, and bound what inversion can reach.

    Prints four things: whether the two schedulers agree about which timestep goes with which
    latent, how well an oracle fed the reverse pass's exact outputs recovers the initial noise,
    how well the ordinary approximation does, and what happens if the model is queried with the
    timestep the training pairs use instead of the one the solver passes.

    Args:
        device: Torch device.
        num_inference_steps: Length of both grids.
        prompt: Caption to condition on.
        seed: Seed for the initial latent.
        model_id: Hub id of the pipeline.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    torch_device = torch.device(device)
    if torch_device.type != "cpu":
        torch.cuda.set_device(torch_device)

    teacher = load_teacher(model_id, torch_device, num_inference_steps, schedule="beta")
    text_audio = teacher.encode_prompt(prompt)
    trajectory, outputs, timesteps = teacher.reverse_trajectory(text_audio, seed)
    logger.info(
        "reverse grid: {} steps, t {:.0f}..{:.0f}", len(timesteps), timesteps[0], timesteps[-1]
    )

    inverse = DPMSolverMultistepInverseScheduler.from_config(teacher.model.scheduler.config)
    inverse.set_timesteps(num_inference_steps, device=torch_device)
    inv_timesteps = [float(t) for t in inverse.timesteps]
    logger.info(
        "inverse grid: {} steps, t {:.0f}..{:.0f}",
        len(inv_timesteps), inv_timesteps[0], inv_timesteps[-1],
    )

    # The training pair for transition i is (trajectory[i + 1], timesteps[i]) -> outputs[i]. The
    # inverse solver walks the same transitions in the opposite order, so its step j corresponds to
    # i = len(timesteps) - 1 - j. These two lists should be identical if the pairing is right.
    trained_t = [timesteps[len(timesteps) - 1 - j] for j in range(len(inv_timesteps))]
    print("\nstep | solver t | trained t | match")
    for j in list(range(3)) + list(range(len(inv_timesteps) - 3, len(inv_timesteps))):
        mark = "yes" if inv_timesteps[j] == trained_t[j] else "NO"
        print(f"{j:>4} | {inv_timesteps[j]:>8.0f} | {trained_t[j]:>9.0f} | {mark}")
    mismatched = sum(a != b for a, b in zip(inv_timesteps, trained_t))
    print(f"\n{mismatched}/{len(inv_timesteps)} steps query a timestep the training pair never saw")

    clean = trajectory[-1:].to(torch_device)
    target = trajectory[:1].to(torch_device)

    # 1. Oracle: feed the exact model output each reverse step used. This is the ceiling -- no
    #    adapter can beat it, because the adapter's whole job is to predict these outputs.
    oracle_latent, oracle_trace = invert(teacher, inverse, clean, text_audio, oracle=list(outputs))
    inverse.set_timesteps(num_inference_steps, device=torch_device)

    # 2. The ordinary approximation the editing code runs.
    plain_latent, plain_trace = invert(teacher, inverse, clean, text_audio)
    inverse.set_timesteps(num_inference_steps, device=torch_device)

    # 3. The approximation, but querying the model at the timestep the training pairs use.
    trained_latent, _ = invert(teacher, inverse, clean, text_audio, timestep_source=trained_t)

    print("\nrecovery of the initial noise (relative L2, lower is better):")
    print(f"  oracle, exact reverse-pass outputs : {relative(oracle_latent, target):.4f}")
    print(f"  plain, solver timesteps            : {relative(plain_latent, target):.4f}")
    print(f"  plain, training-pair timesteps     : {relative(trained_latent, target):.4f}")

    # Does the inverse pass actually visit the reverse trajectory's latents? If it drifts, the
    # cached pairs describe inputs the adapter never sees at inference.
    print("\nlatent the solver feeds at step j vs the reverse trajectory's latent there:")
    for entry in [oracle_trace[0], oracle_trace[1], oracle_trace[len(oracle_trace) // 2],
                  oracle_trace[-1]]:
        j = entry["step"]
        expected = trajectory[len(timesteps) - j : len(timesteps) - j + 1]
        print(
            f"  step {j:>3} tau={entry['tau']:>6.0f}  "
            f"||fed - trajectory[{len(timesteps) - j}]|| = "
            f"{relative(entry['latent'], expected):.4f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
