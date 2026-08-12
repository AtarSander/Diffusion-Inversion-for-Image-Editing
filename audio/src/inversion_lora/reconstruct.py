# ABOUTME: Invert-then-denoise reconstruction eval for the AudioLDM2 inversion LoRA, scoring latent
# ABOUTME: MSE and mel MSE/SSIM/PSNR on both generated latents and real audio crops.

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_inversion.ddim_inversion import next_step  # noqa: E402

from src.inversion_lora.generate_trajectories import latent_height  # noqa: E402

Conditioning = dict[str, torch.Tensor]

MEL_FRAMES_PER_SECOND = 100


def encode_prompts(ldm, prompts: list[str]) -> Conditioning:
    """Encode a batch of prompts into the three AudioLDM2 conditioning tensors.

    Args:
        ldm: `AudioLDM2Wrapper`.
        prompts: One caption per batch element.

    Returns:
        `encoder_hidden_states`, `class_labels` and `encoder_attention_mask`, batched.
    """
    hidden, t5_embeds, t5_mask = ldm.encode_text(prompts)
    assert hidden.shape[0] == len(prompts), (hidden.shape, len(prompts))
    return {
        "encoder_hidden_states": hidden,
        "class_labels": t5_embeds,
        "encoder_attention_mask": t5_mask,
    }


@torch.no_grad()
def ddim_invert(ldm, x0: torch.Tensor, cond: Conditioning) -> torch.Tensor:
    """Invert a clean latent to the noise end of the DDIM schedule, conditionally, without CFG.

    Each step evaluates epsilon at the *current, cleaner* latent while passing the *noisier*
    timestep. That is precisely the approximation the inversion LoRA is trained to correct
    (eps_phi(x_{t-1}, t) ~ eps_theta(x_t, t)), so enabling the adapter here is what makes the
    inversion near-exact. The vendored `next_step` supplies the schedule algebra unchanged.

    Args:
        ldm: `AudioLDM2Wrapper` with the adapter in whatever state the caller wants measured.
        x0: Clean latent `[B, C, H, W]`.
        cond: Output of `encode_prompts`, batched to match `x0`.

    Returns:
        Latent at the noisiest timestep, `[B, C, H, W]`.
    """
    timesteps = ldm.model.scheduler.timesteps
    latent = x0.clone().detach()
    for i in range(len(timesteps)):
        t = timesteps[len(timesteps) - i - 1]
        model_input = ldm.model.scheduler.scale_model_input(latent, t)
        eps = ldm.unet_forward(model_input, timestep=t, **cond)[0].sample
        latent = next_step(ldm, eps, t, latent)
    return latent


@torch.no_grad()
def ddim_denoise(ldm, x_t: torch.Tensor, cond: Conditioning) -> torch.Tensor:
    """Denoise back to a clean latent along the same schedule, conditionally, without CFG.

    Args:
        ldm: `AudioLDM2Wrapper`.
        x_t: Latent at the noisiest timestep, `[B, C, H, W]`.
        cond: Output of `encode_prompts`, batched to match `x_t`.

    Returns:
        Reconstructed clean latent `[B, C, H, W]`.
    """
    latent = x_t.clone().detach()
    for t in ldm.model.scheduler.timesteps:
        model_input = ldm.model.scheduler.scale_model_input(latent, t)
        eps = ldm.unet_forward(model_input, timestep=t, **cond)[0].sample
        latent = ldm.model.scheduler.step(eps, t, latent, eta=0).prev_sample
    return latent


def mel_metrics(mel_ref: torch.Tensor, mel_rec: torch.Tensor) -> dict[str, float]:
    """Per-example mel MSE, SSIM and PSNR, averaged over the batch.

    SSIM and PSNR come from scikit-image with the same `data_range` convention the editing
    benchmark uses in `src/metrics/alignment.py`, so the numbers mean the same thing here.

    Args:
        mel_ref: Reference mel `[B, 1, T, F]`.
        mel_rec: Reconstructed mel, same shape.

    Returns:
        Mean `mel_mse`, `mel_ssim` and `mel_psnr`.
    """
    assert mel_ref.shape == mel_rec.shape, (mel_ref.shape, mel_rec.shape)
    reference = mel_ref.squeeze(1).float().cpu().numpy()
    reconstruction = mel_rec.squeeze(1).float().cpu().numpy()
    assert reference.ndim == 3, reference.shape

    scores: dict[str, list[float]] = {"mel_mse": [], "mel_ssim": [], "mel_psnr": []}
    for one_ref, one_rec in zip(reference, reconstruction):
        data_range = max(one_ref.max(), one_rec.max()) - min(one_ref.min(), one_rec.min())
        scores["mel_mse"].append(float(np.mean((one_ref - one_rec) ** 2)))
        scores["mel_ssim"].append(float(structural_similarity(one_ref, one_rec, data_range=data_range)))
        scores["mel_psnr"].append(float(peak_signal_noise_ratio(one_ref, one_rec, data_range=data_range)))
    return {key: float(np.mean(values)) for key, values in scores.items()}


@torch.no_grad()
def reconstruction_metrics(
    ldm,
    x0: torch.Tensor,
    prompts: list[str],
    batch_size: int,
    set_lora_enabled: Callable[[bool], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Invert then denoise each latent and score how well the round trip returns it.

    Inversion runs with the adapter in its current state; the reverse pass always runs with the
    adapter disabled, since the LoRA's job is to fix the inversion direction and the denoiser
    being reconstructed is the frozen teacher.

    Args:
        ldm: `AudioLDM2Wrapper`.
        x0: Clean latents to round-trip, `[N, C, H, W]`.
        prompts: One caption per latent.
        batch_size: Latents per forward pass.
        set_lora_enabled: Toggle for the adapter. `None` measures plain DDIM inversion.
        progress: Optional callback for a one-line status per batch.

    Returns:
        `latent_mse` plus the mel metrics averaged over all N examples, and the inverted latents
        `[N, C, H, W]` on the CPU so the caller can run the distributional noise checks on them.
    """
    assert x0.shape[0] == len(prompts), (x0.shape, len(prompts))
    latent_mse: list[float] = []
    mel_scores: list[dict[str, float]] = []
    weights: list[int] = []
    inverted: list[torch.Tensor] = []

    for start in range(0, x0.shape[0], batch_size):
        reference = x0[start : start + batch_size].to(ldm.device)
        cond = encode_prompts(ldm, prompts[start : start + batch_size])

        if set_lora_enabled is not None:
            set_lora_enabled(True)
        x_t = ddim_invert(ldm, reference, cond)
        if set_lora_enabled is not None:
            set_lora_enabled(False)
        try:
            reconstruction = ddim_denoise(ldm, x_t, cond)
        finally:
            if set_lora_enabled is not None:
                set_lora_enabled(True)

        assert reconstruction.shape == reference.shape, (reconstruction.shape, reference.shape)
        latent_mse.append(float(torch.mean((reconstruction.float() - reference.float()) ** 2)))
        mel_scores.append(mel_metrics(ldm.vae_decode(reference), ldm.vae_decode(reconstruction)))
        weights.append(reference.shape[0])
        inverted.append(x_t.detach().cpu())
        if progress is not None:
            progress(
                f"reconstruction {start + reference.shape[0]}/{x0.shape[0]}: "
                f"latent_mse={latent_mse[-1]:.5f} mel_psnr={mel_scores[-1]['mel_psnr']:.2f}"
            )

    total = sum(weights)
    out = {"latent_mse": sum(m * w for m, w in zip(latent_mse, weights)) / total}
    for key in mel_scores[0]:
        out[key] = sum(s[key] * w for s, w in zip(mel_scores, weights)) / total
    return out, torch.cat(inverted, dim=0)


@torch.no_grad()
def generate_eval_latents(
    ldm, prompts: list[str], seed: int, batch_size: int, duration_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample clean latents from the frozen teacher, to be round-tripped later.

    These stand in for "generated samples": the teacher's own output distribution, which the
    inversion should be able to recover exactly. The initial noise is returned too, because for
    generated audio it is the exact reference the inverted latent should match.

    Args:
        ldm: `AudioLDM2Wrapper`.
        prompts: One caption per latent.
        seed: Seed for the initial noise; fixed so every eval scores the same audio.
        batch_size: Latents per forward pass.
        duration_s: Audio duration; must match the real-audio crop so both sets share a shape.

    Returns:
        Clean latents and the initial noise that produced them, both `[N, C, H, W]` on CPU.
    """
    pipe = ldm.model
    # prepare_latents takes the mel height and divides by the VAE scale factor itself, so this
    # must be the mel grid, not the latent grid. Shared with the trajectory generator so the two
    # cannot disagree about what a 10.24 s clip is.
    height = latent_height(pipe, duration_s)
    generated: list[torch.Tensor] = []
    initial_noise: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        latents = pipe.prepare_latents(
            batch_size=len(chunk),
            num_channels_latents=pipe.unet.config.in_channels,
            height=height,
            dtype=pipe.unet.dtype,
            device=torch.device(ldm.device),
            generator=torch.Generator(device=ldm.device).manual_seed(seed + start),
            latents=None,
        )
        initial_noise.append(latents.detach().cpu())
        cond = encode_prompts(ldm, chunk)
        generated.append(ddim_denoise(ldm, latents, cond).cpu())
    return torch.cat(generated, dim=0), torch.cat(initial_noise, dim=0)


def held_out_prompts(data_root: str | Path, sample_ids: set[int], count: int) -> list[str]:
    """Read the captions of validation trajectories, so the eval prompts are never trained on.

    Args:
        data_root: Trajectory dataset root.
        sample_ids: Validation `sample_idx` values, from the same split the trainer uses.
        count: How many captions to take, in sample-index order.

    Returns:
        `count` captions.
    """
    import json

    prompts: list[str] = []
    for sample_dir in sorted(Path(data_root).glob("sample_*")):
        if len(prompts) >= count:
            break
        meta = json.loads((sample_dir / "meta.json").read_text())
        if int(meta["sample_idx"]) not in sample_ids:
            continue
        prompts.append(str(json.loads((sample_dir / "prompt.json").read_text())["prompt"]))
    assert len(prompts) == count, (
        f"only {len(prompts)} held-out prompts available for {count} eval samples; "
        "raise val_fraction or lower recon_num_generated"
    )
    return prompts


def real_audio_fixtures(
    prompts_csv: str | Path, audio_root: str | Path, count: int, seed: int
) -> tuple[list[Path], list[str]]:
    """Pick real MedleyDB mixes and their source captions for the real-audio reconstruction.

    Inverting real audio needs a source prompt; the MedleyMD caption for that track is the
    honest choice, since it describes the audio actually being inverted.

    Args:
        prompts_csv: MedleyMD prompt CSV.
        audio_root: Root holding `<Track>/<Track>_MIX.wav`.
        count: How many crops to score.
        seed: Seed for the track choice.

    Returns:
        Audio paths and their source captions, one per requested crop.
    """
    import pandas as pd

    frame = pd.read_csv(prompts_csv, index_col=0).drop_duplicates(subset="filename")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(frame), size=count, replace=count > len(frame))

    paths: list[Path] = []
    captions: list[str] = []
    for position in chosen:
        row = frame.iloc[int(position)]
        paths.append(Path(audio_root) / str(row["filename"]).split("_MIX")[0] / str(row["filename"]))
        captions.append(str(row["source_captions"]))
    return paths, captions


def load_real_latents(
    ldm, audio_paths: list[Path], seed: int, duration_s: float
) -> tuple[torch.Tensor, list[str]]:
    """Encode fixed random crops of real audio into latents.

    Args:
        ldm: `AudioLDM2Wrapper`.
        audio_paths: One file per requested crop; repeats are expected when the pool is small.
        seed: Seed for the crop offsets, so the same excerpts are scored at every eval.
        duration_s: Crop length in seconds.

    Returns:
        Latents `[N, C, H, W]` on the CPU, and the crop identifiers for logging.
    """
    from utils import load_audio

    fn_stft = ldm.get_fn_STFT()
    # TacotronSTFT runs at 16 kHz with hop 160, so the mel grid is exactly 100 frames a second.
    frames = int(duration_s * MEL_FRAMES_PER_SECOND)
    rng = np.random.default_rng(seed)
    latents: list[torch.Tensor] = []
    names: list[str] = []

    for path in audio_paths:
        mel, _, _ = load_audio(
            str(path), fn_stft, device=ldm.device, stft=True, model_sr=ldm.get_sr()
        )
        assert mel.ndim == 4, f"expected [1, 1, T, F] mel, got {tuple(mel.shape)}"
        assert mel.shape[2] > frames, f"{path} is shorter than {duration_s}s"
        offset = int(rng.integers(0, mel.shape[2] - frames))
        latents.append(ldm.vae_encode(mel[:, :, offset : offset + frames, :]).cpu())
        names.append(f"{path.stem}@{offset / MEL_FRAMES_PER_SECOND:.1f}s")

    return torch.cat(latents, dim=0), names
