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

from src.inversion_lora.generate_trajectories import latent_height  # noqa: E402
from src.inversion_lora.noise_metrics import noise_report  # noqa: E402
from src.inversion_lora.reconstruct import (  # noqa: E402
    batch_latents,
    crop_to_window,
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
    real_max_duration_s: float | None = None,
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
        duration_s: Generated-audio length, and the real crop length when the window is fixed.
        real_max_duration_s: Cap for the natural-length real window; `None` keeps the fixed crop.
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
    real, names = load_real_latents(
        ldm,
        audio_paths,
        seed=seed,
        duration_s=duration_s,
        max_duration_s=real_max_duration_s,
    )
    logger.info("real {} from {}", [tuple(latent.shape) for latent in real], names)

    window = latent_height(ldm.model, duration_s) // ldm.model.vae_scale_factor

    for kind, latents, prompt_list, reference in (
        ("generated", list(generated.split(1)), prompts, initial_noise),
        ("real", real, audio_prompts, None),
    ):  # generated audio has its true initial noise; real audio can only get a stand-in draw
        scores, inverted = reconstruction_metrics(
            ldm, batch_latents(latents, batch_size), prompt_list, progress=logger.info
        )
        logger.info("{}: {}", kind, scores)
        inverted_window = crop_to_window(inverted, window)
        report = noise_report(
            inverted_window,
            reference if reference is not None else torch.randn(inverted_window.shape),
            prefix=kind,
            reference_is_ground_truth=kind == "generated",
            seed=seed,
        )
        logger.info("{}: {}", kind, report)

    logger.success("Reconstruction eval wiring is intact")


if __name__ == "__main__":
    fire.Fire(main)
