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

    Both come from scikit-image over an empirical `data_range` -- about 12.6 on VAE-decoded
    log-mels -- which matches how `src/metrics/alignment.py` scores SSIM but not how it scores
    PSNR. Mel PSNR here is NOT comparable with mel PSNR from the editing benchmark, for two
    independent reasons, so only ever compare within one path:

    1. What is compared. Here both sides are VAE decodes of a latent, so decoder error cancels
       and no vocoder runs. The benchmark pairs two *wav files*, so its number also carries VAE
       encode+decode, the 16 kHz HiFi-GAN vocoder and an STFT re-analysis.
    2. How the mel is scaled. `audioldm_eval` re-analyses each wav (first channel only, 16 kHz,
       DC removed) into `clip((20 * log10(mel) + 80) / 100, 0, 1)`, mapping -80..+20 dB onto
       [0, 1] and clipping outside it. Its `psnr()` call passes no `data_range`, which is right
       for that representation -- scikit-image's float fallback of 1.0 *is* its full scale, the
       same convention as 255 for uint8. The empirical per-file range used here is the less
       standard of the two. Either way the scalings differ, so no fixed dB offset relates the
       numbers.

    The benchmark also truncates both mels to the shorter clip from frame 0, so it assumes the
    output is sample-aligned with the input.

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


def batch_latents(latents: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    """Stack consecutive equal-shaped latents into batches of at most `batch_size`.

    Real-audio fixtures keep each track's own length, so they are a list of differently shaped
    latents rather than one tensor. Only same-shaped neighbours can share a forward pass, so a
    run of distinct shapes degrades to batch 1 -- which costs nothing at these lengths, because
    a single 60 s latent already saturates the GPU (measured: per-sample forward time is flat in
    batch size at 60 s, while at 10.24 s batch 8 is 2.9x more efficient per sample).

    Args:
        latents: Latents `[n_i, C, H_i, W]`, in the order they should be scored.
        batch_size: Most examples per batch.

    Returns:
        Batches `[n, C, H, W]`, preserving order and total example count.
    """
    assert latents, "no latents to batch"
    assert batch_size > 0, batch_size

    batches: list[torch.Tensor] = []
    group: list[torch.Tensor] = []
    for latent in latents:
        assert latent.ndim == 4, f"expected [n, C, H, W], got {tuple(latent.shape)}"
        full = sum(item.shape[0] for item in group) + latent.shape[0] > batch_size
        if group and (latent.shape[1:] != group[0].shape[1:] or full):
            batches.append(torch.cat(group, dim=0))
            group = []
        group.append(latent)
    batches.append(torch.cat(group, dim=0))

    assert sum(b.shape[0] for b in batches) == sum(t.shape[0] for t in latents)
    return batches


def crop_to_window(latents: list[torch.Tensor], height: int) -> torch.Tensor:
    """Cut every latent to the same leading `height` frames and stack them.

    The distributional noise checks estimate per-dimension statistics across the batch, so they
    need one shape for the whole set, which variable-length crops do not have. Cutting to a fixed
    window rather than to the shortest fixture also keeps those numbers comparable across runs
    whose crops differ.

    Args:
        latents: Latents `[n_i, C, H_i, W]`, each at least `height` frames tall.
        height: Latent frames to keep.

    Returns:
        Stacked latents `[N, C, height, W]`.
    """
    assert latents, "no latents to crop"
    for latent in latents:
        assert latent.shape[2] >= height, (
            f"latent of height {latent.shape[2]} is shorter than the {height}-frame noise window"
        )
    return torch.cat([latent[:, :, :height, :] for latent in latents], dim=0)


@torch.no_grad()
def reconstruction_metrics(
    ldm,
    batches: list[torch.Tensor],
    prompts: list[str],
    set_lora_enabled: Callable[[bool], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    """Invert then denoise each latent and score how well the round trip returns it.

    Inversion runs with the adapter in its current state; the reverse pass always runs with the
    adapter disabled, since the LoRA's job is to fix the inversion direction and the denoiser
    being reconstructed is the frozen teacher.

    Args:
        ldm: `AudioLDM2Wrapper`.
        batches: Clean latents to round-trip, grouped by `batch_latents`.
        prompts: One caption per latent, flat across all batches.
        set_lora_enabled: Toggle for the adapter. `None` measures plain DDIM inversion.
        progress: Optional callback for a one-line status per batch.

    Returns:
        `latent_mse` plus the mel metrics averaged over every example, and the inverted latents
        per batch on the CPU, which the caller feeds to the distributional noise checks. They are
        returned unstacked because the batches need not share a shape.
    """
    total = sum(batch.shape[0] for batch in batches)
    assert total == len(prompts), (total, len(prompts))
    latent_mse: list[float] = []
    mel_scores: list[dict[str, float]] = []
    weights: list[int] = []
    inverted: list[torch.Tensor] = []

    scored = 0
    for batch in batches:
        reference = batch.to(ldm.device)
        cond = encode_prompts(ldm, prompts[scored : scored + batch.shape[0]])
        scored += batch.shape[0]

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
                f"reconstruction {scored}/{total} at latent {tuple(reference.shape[2:])}: "
                f"latent_mse={latent_mse[-1]:.5f} mel_psnr={mel_scores[-1]['mel_psnr']:.2f}"
            )

    assert scored == total, (scored, total)
    out = {"latent_mse": sum(m * w for m, w in zip(latent_mse, weights)) / total}
    for key in mel_scores[0]:
        out[key] = sum(s[key] * w for s, w in zip(mel_scores, weights)) / total
    return out, inverted


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
    ldm,
    audio_paths: list[Path],
    seed: int,
    duration_s: float,
    max_duration_s: float | None = None,
) -> tuple[list[torch.Tensor], list[str]]:
    """Encode crops of real audio into latents, one per file.

    Two windows are available. The default takes a fixed `duration_s` crop at a random offset,
    which gives every latent the same shape. Setting `max_duration_s` instead takes each track at
    its own length, capped there and measured from the start -- exactly what
    `create_truncated_audio` feeds the editing pipeline, so the reconstruction is scored at the
    geometry the edits actually run on. Crops then differ in shape, which is why this returns a
    list rather than a stacked tensor.

    Args:
        ldm: `AudioLDM2Wrapper`.
        audio_paths: One file per requested crop; repeats are expected when the pool is small.
        seed: Seed for the crop offsets, so the same excerpts are scored at every eval. Unused in
            the natural-length window, which has no offset to choose.
        duration_s: Crop length in seconds, for the fixed window.
        max_duration_s: Cap for the natural-length window; `None` selects the fixed window.

    Returns:
        Latents `[1, C, H_i, W]` on the CPU, one per path, and the crop identifiers for logging.
    """
    from utils import load_audio

    fn_stft = ldm.get_fn_STFT()
    # TacotronSTFT runs at 16 kHz with hop 160, so the mel grid is exactly 100 frames a second.
    scale = ldm.model.vae_scale_factor
    rng = np.random.default_rng(seed)
    latents: list[torch.Tensor] = []
    names: list[str] = []

    for path in audio_paths:
        mel, _, _ = load_audio(
            str(path), fn_stft, device=ldm.device, stft=True, model_sr=ldm.get_sr()
        )
        assert mel.ndim == 4, f"expected [1, 1, T, F] mel, got {tuple(mel.shape)}"
        if max_duration_s is None:
            frames = int(duration_s * MEL_FRAMES_PER_SECOND)
            assert mel.shape[2] > frames, f"{path} is shorter than {duration_s}s"
            offset = int(rng.integers(0, mel.shape[2] - frames))
        else:
            # The VAE downsamples the mel by vae_scale_factor, so a natural length has to be
            # trimmed to a multiple of it; the fixed window is already one by construction.
            frames = min(mel.shape[2], int(max_duration_s * MEL_FRAMES_PER_SECOND))
            frames -= frames % scale
            offset = 0
        assert frames > 0, f"{path} yielded no usable frames"
        latents.append(ldm.vae_encode(mel[:, :, offset : offset + frames, :]).cpu())
        names.append(
            f"{path.stem}@{offset / MEL_FRAMES_PER_SECOND:.1f}s"
            f"+{frames / MEL_FRAMES_PER_SECOND:.2f}s"
        )

    return latents, names
