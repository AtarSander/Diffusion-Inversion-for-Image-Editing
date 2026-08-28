# ABOUTME: First vs second order ODE inversion on Stable Audio at matched NFE: exactness of the
# ABOUTME: inverse, round-trip error under substitution, and the sampling quality of each order.

import json
import subprocess
import sys
from pathlib import Path

import fire
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
    ExactDPMSolver,
    StableAudioTeacher,
    decode_to_audio,
    load_teacher,
)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance."""
    return float((a - b).norm() / b.norm())


@torch.no_grad()
def sample(
    teacher: StableAudioTeacher, solver: ExactDPMSolver, text_audio: torch.Tensor, seed: int,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with the given order, keeping every latent and every data prediction.

    Step 0 is always first order: a multistep update needs a previous prediction, and there is
    none yet, which is what diffusers' `lower_order_nums` bookkeeping expresses.

    Args:
        teacher: The loaded teacher.
        solver: Solver over the grid.
        text_audio: Cross-attention states.
        seed: Seed for the initial latent.
        order: 1 or 2.

    Returns:
        `(trajectory, data)`, both on the CPU. One model call per step either way.
    """
    shape = (1, teacher.pipe.transformer.config.in_channels, teacher.latent_length)
    x = torch.randn(
        shape, generator=torch.Generator(device=teacher.device).manual_seed(seed),
        device=teacher.device,
    ) * solver.sigmas[0]

    trajectory, data = [x.cpu()], []
    for index in tqdm(range(len(solver.timesteps)), desc=f"order {order}", leave=False):
        t = torch.tensor([solver.timesteps[index]], device=teacher.device)
        raw = teacher.forward(solver.model_input(x, index), t, text_audio)
        prediction = solver.data_prediction(x, raw, index)
        # First order at step 0 (no history) and at the final step, where sigma_t = 0 makes
        # r0 = h_0 / h vanish and the D1 term divide by zero. diffusers calls this
        # `lower_order_final`.
        if order == 2 and 1 <= index < solver.invertible_steps:
            x = solver.forward_2nd(x, prediction, data[-1].to(teacher.device), index)
        else:
            x = solver.forward(x, prediction, index)
        data.append(prediction.cpu())
        trajectory.append(x.cpu())
    return torch.cat(trajectory), torch.cat(data)


@torch.no_grad()
def invert(
    teacher: StableAudioTeacher, solver: ExactDPMSolver, trajectory: torch.Tensor,
    data: torch.Tensor, text_audio: torch.Tensor, order: int, oracle: bool,
) -> torch.Tensor:
    """Walk the grid backwards with the exact inverse of the same order.

    With `oracle`, each step is fed the predictions the reverse pass actually consumed, which is
    the ceiling: an exact inverse must then return the initial noise. Otherwise the prediction at
    the noisier point is substituted by one at the cleaner latent, and for second order the
    *previous* step's prediction is substituted by the one computed at the previous inverse step --
    one grid point too clean, since inversion has not reached the noisier one yet.

    Args:
        teacher: The loaded teacher.
        solver: Solver over the grid.
        trajectory: Latents from `sample`.
        data: Data predictions from `sample`.
        text_audio: Cross-attention states.
        order: 1 or 2.
        oracle: Feed the true predictions instead of substituting.

    Returns:
        The recovered noisy latent.
    """
    start = solver.invertible_steps
    x = trajectory[start : start + 1].to(teacher.device)
    previous = None
    for index in reversed(range(start)):
        if oracle:
            current = data[index : index + 1].to(teacher.device)
            earlier = data[index - 1 : index].to(teacher.device) if index >= 1 else None
        else:
            t = torch.tensor([solver.timesteps[index]], device=teacher.device)
            raw = teacher.forward(solver.model_input(x, index), t, text_audio)
            current = solver.data_prediction(x, raw, index)
            earlier = previous
        if order == 2 and index >= 1 and earlier is not None:
            x = solver.inverse_2nd(x, current, earlier, index)
        else:
            x = solver.inverse(x, current, index)
        previous = current
    return x


def main(
    device: str = "cuda:0",
    num_inference_steps: int = 100,
    prompt: str = "A gentle acoustic guitar melody with soft percussion.",
    seed: int = 42,
    output_dir: str = "output/sao_orders",
    model_id: str = MODEL_ID,
) -> None:
    """Score both orders on inverse exactness, substituted round trip, and audio.

    Both orders cost one model call per step, so every number here is at matched NFE.

    Args:
        device: Torch device.
        num_inference_steps: Length of the grid.
        prompt: Caption to sample from.
        seed: Seed for the initial latent.
        output_dir: Destination for wavs and JSON, relative to `audio/`.
        model_id: Hub id of the pipeline.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    torch_device = torch.device(device)
    if torch_device.type != "cpu":
        torch.cuda.set_device(torch_device)
    out_dir = AUDIO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(model_id, torch_device, num_inference_steps, schedule="cosine")
    solver = ExactDPMSolver(teacher.model.scheduler)
    text_audio = teacher.encode_prompt(prompt)

    report = {}
    for order in (1, 2):
        trajectory, data = sample(teacher, solver, text_audio, seed, order)
        target = trajectory[:1]
        oracle = invert(teacher, solver, trajectory, data, text_audio, order, oracle=True).cpu()
        substituted = invert(
            teacher, solver, trajectory, data, text_audio, order, oracle=False
        ).cpu()
        audio = decode_to_audio(teacher, trajectory[-1:])[0]
        wav = out_dir / f"order{order}_{num_inference_steps}steps.wav"
        torchaudio.save(str(wav), audio, sample_rate=teacher.pipe.vae.config.sampling_rate)

        report[f"order{order}"] = {
            "nfe_sampling": len(solver.timesteps),
            "nfe_inversion": solver.invertible_steps,
            "oracle_round_trip": relative(oracle, target),
            "substituted_round_trip": relative(substituted, target),
            "audio_rms": float(audio.pow(2).mean().sqrt()),
            "audio_peak": float(audio.abs().max()),
            "final_latent_rms": float(trajectory[-1].pow(2).mean().sqrt()),
            "wav": str(wav.relative_to(AUDIO_ROOT)),
        }
        r = report[f"order{order}"]
        print(f"\norder {order}: {r['nfe_sampling']} NFE sampling, {r['nfe_inversion']} inversion")
        print(f"  oracle round trip      : {r['oracle_round_trip']:.6f}")
        print(f"  substituted round trip : {r['substituted_round_trip']:.6f}")
        print(f"  audio RMS {r['audio_rms']:.4f} peak {r['audio_peak']:.4f} -> {wav.name}")

    ratio = (report["order2"]["substituted_round_trip"]
             / report["order1"]["substituted_round_trip"])
    print(f"\nsecond order substituted round trip is {ratio:.2f}x the first order's")

    path = out_dir / f"orders_{num_inference_steps}steps.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=AUDIO_ROOT, text=True
                ).strip(),
                "prompt": prompt, "seed": seed,
                "num_inference_steps": num_inference_steps, "orders": report,
            },
            f, indent=2,
        )
    logger.success("Wrote {}", path)


if __name__ == "__main__":
    fire.Fire(main)
