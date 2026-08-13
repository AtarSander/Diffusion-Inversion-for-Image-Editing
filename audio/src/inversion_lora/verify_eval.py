# ABOUTME: Verify the reconstruction eval measures what it claims: an untrained LoRA is an exact
# ABOUTME: identity so it must reproduce the no-LoRA baseline, and a perturbed one must move it.

import sys
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models import load_model  # noqa: E402

from src.inversion_lora.reconstruct import real_audio_fixtures  # noqa: E402
from src.inversion_lora.train import AudioLDM2InversionTrainer, NullTracker  # noqa: E402


def strip_prefix(metrics: dict[str, float]) -> dict[str, float]:
    """Drop the leading `eval/` or `eval_no_lora/` so the two passes can be compared key by key."""
    return {key.split("/", 1)[1]: value for key, value in metrics.items()}


def main(
    config_name: str = "train_inversion_lora_localtest",
    device: str = "cuda:0",
    num_inference_steps: int | None = None,
    num_samples: int = 2,
) -> None:
    """Check the eval is sensitive to the adapter and to nothing else.

    Two properties are asserted:

    1. With `init_lora_weights=true` the B matrices are zero, so the adapter is mathematically
       the identity. Running the eval with it enabled must reproduce the no-LoRA baseline
       exactly. A difference means the toggle leaks, or the eval is not deterministic.
    2. After writing non-zero values into the B matrices the eval must change. If it does not,
       the adapter is not reaching the inversion pass and every training curve would be flat
       for reasons that have nothing to do with learning.

    Args:
        config_name: Training config to borrow LoRA and eval settings from.
        device: Torch device.
        num_inference_steps: Override the DDIM grid; shorter is faster and equally valid here.
        num_samples: Generated and real samples per pass.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    cfg = OmegaConf.load(AUDIO_ROOT / f"config/{config_name}.yaml")
    base = OmegaConf.load(AUDIO_ROOT / "config/train_inversion_lora.yaml")
    cfg = OmegaConf.merge(base, cfg)
    cfg.recon_num_generated = num_samples
    cfg.recon_num_real = num_samples
    cfg.recon_batch_size = num_samples
    if num_inference_steps is not None:
        cfg.num_inference_steps = num_inference_steps

    ldm = load_model(str(cfg.model_id), device, int(cfg.num_inference_steps), edit_method="ddim")
    trainer = AudioLDM2InversionTrainer(ldm, cfg, NullTracker())

    audio_paths, audio_prompts = real_audio_fixtures(
        cfg.recon_prompts_csv, cfg.recon_audio_root, count=num_samples, seed=int(cfg.seed)
    )
    prompts = ["a bright acoustic guitar melody", "a slow ambient synth pad"][:num_samples]
    trainer.prepare_reconstruction_fixtures(prompts, audio_paths, audio_prompts)
    baseline = strip_prefix(trainer.baseline_reconstruction)

    identity = strip_prefix(trainer.reconstruction_eval())
    mismatched = {
        key: (baseline[key], identity[key])
        for key in baseline
        if baseline[key] != identity[key]
    }
    logger.info("baseline  : {}", {k: round(v, 6) for k, v in baseline.items()})
    logger.info("untrained : {}", {k: round(v, 6) for k, v in identity.items()})
    assert not mismatched, (
        "An untrained LoRA is the identity, so the eval must reproduce the baseline exactly. "
        f"These differ: {mismatched}"
    )
    logger.success("Untrained LoRA reproduces the no-LoRA baseline exactly ({} metrics)", len(baseline))

    perturbed_tensors = 0
    with torch.no_grad():
        for name, param in trainer.unet.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.01)
                perturbed_tensors += 1
    assert perturbed_tensors > 0, "No lora_B parameters found to perturb"
    logger.info("Perturbed {} lora_B tensors", perturbed_tensors)

    perturbed = strip_prefix(trainer.reconstruction_eval())
    logger.info("perturbed : {}", {k: round(v, 6) for k, v in perturbed.items()})
    moved = [key for key in baseline if baseline[key] != perturbed[key]]
    assert "generated/latent_mse" in moved, (
        "Perturbing the adapter left the reconstruction error unchanged, so the LoRA is not "
        "being applied during inversion."
    )
    logger.success(
        "Perturbing the adapter moved {}/{} metrics, including generated/latent_mse "
        "({:.6f} -> {:.6f})",
        len(moved),
        len(baseline),
        baseline["generated/latent_mse"],
        perturbed["generated/latent_mse"],
    )


if __name__ == "__main__":
    fire.Fire(main)
