# ABOUTME: Fast end-to-end check of the inversion-LoRA reconstruction eval: builds tiny generated
# ABOUTME: and real fixtures, round-trips them on a short DDIM grid, and prints every metric.

import sys
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models import load_model  # noqa: E402

from src.inversion_lora.noise_metrics import noise_report  # noqa: E402
from src.inversion_lora.reconstruct import (  # noqa: E402
    generate_eval_latents,
    load_real_latents,
    real_audio_fixtures,
    reconstruction_metrics,
)


def main(
    device: str = "cuda:0",
    num_inference_steps: int = 20,
    num_samples: int = 2,
    batch_size: int = 2,
    duration_s: float = 10.24,
    seed: int = 42,
) -> None:
    """Run the reconstruction eval end to end on a handful of samples.

    Everything here is the code the periodic training eval calls, only with a short schedule, so
    a pass proves the wiring rather than the quality. With no LoRA injected the round trip is
    plain DDIM inversion, whose error is exactly the gap the LoRA has to close.

    Args:
        device: Torch device.
        num_inference_steps: DDIM grid length; the real eval uses 200.
        num_samples: Generated and real samples to score.
        batch_size: Samples per forward pass.
        duration_s: Real-audio crop length.
        seed: Seed for fixtures.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    import os

    ldm = load_model("cvssp/audioldm2-large", device, num_inference_steps, edit_method="ddim")

    prompts = ["a bright acoustic guitar melody", "a slow ambient synth pad"][:num_samples]
    generated, initial_noise = generate_eval_latents(
        ldm, prompts, seed=seed, batch_size=batch_size, duration_s=duration_s
    )
    logger.info("generated {} noise {}", tuple(generated.shape), tuple(initial_noise.shape))

    audio_paths, audio_prompts = real_audio_fixtures(
        os.environ["MEDLEY_PROMPTS_CSV"], os.environ["MEDLEYDB_AUDIO_DIR"], num_samples, seed
    )
    real, names = load_real_latents(ldm, audio_paths, seed=seed, duration_s=duration_s)
    logger.info("real {} from {}", tuple(real.shape), names)
    assert real.shape[1:] == generated.shape[1:], (real.shape, generated.shape)

    for kind, latents, prompt_list, reference in (
        ("generated", generated, prompts, initial_noise),
        ("real", real, audio_prompts, torch.randn(real.shape)),
    ):  # generated audio has its true initial noise; real audio can only get a stand-in draw
        scores, inverted = reconstruction_metrics(
            ldm, latents, prompt_list, batch_size=batch_size, progress=logger.info
        )
        logger.info("{}: {}", kind, scores)
        report = noise_report(
            inverted, reference, prefix=kind, reference_is_ground_truth=kind == "generated", seed=seed
        )
        logger.info("{}: {}", kind, report)

    logger.success("Reconstruction eval wiring is intact")


if __name__ == "__main__":
    fire.Fire(main)
