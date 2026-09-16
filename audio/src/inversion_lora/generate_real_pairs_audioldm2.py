# ABOUTME: Build inversion-LoRA training pairs from REAL audio for AudioLDM2: forward-noise each
# ABOUTME: clip's VAE latent to every DDIM timestep, teacher-predict epsilon, form exact-DDIM-step
# ABOUTME: inputs with the shifted-timestep pairing the AudioLDM2 adapter is deployed under.

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_EDITING_CODE = AUDIO_ROOT / "editing/AudioEditingCode/code"
for p in (AUDIO_ROOT, AUDIO_EDITING_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import load_model
from src.inversion_lora.generate_trajectories import (
    git_sha,
    latent_height,
)
from utils import load_audio


def encode_clip(ldm, path: Path, target_frames: int) -> torch.Tensor:
    """Encode one real clip to an AudioLDM2 VAE latent at a fixed mel length.

    Mel frames run at 100/s (TacotronSTFT, 16 kHz, hop 160). A clip shorter than the target is
    padded with its own floor (silence) and a longer one is cropped from the start, so every
    latent has the model's native shape and matches the trajectory dataset.

    Args:
        ldm: `AudioLDM2Wrapper`.
        path: Clip path.
        target_frames: Mel frames to produce (latent_height * vae_scale_factor).

    Returns:
        Latent `[1, C, H, W]` on the CPU.
    """
    fn_stft = ldm.get_fn_STFT()
    mel, _, _ = load_audio(str(path), fn_stft, device=ldm.device, stft=True, model_sr=ldm.get_sr())
    assert mel.ndim == 4, f"expected [1, 1, T, F] mel, got {tuple(mel.shape)}"
    assert target_frames % ldm.model.vae_scale_factor == 0, target_frames
    frames = mel.shape[2]
    if frames < target_frames:
        pad = mel.new_full((1, 1, target_frames - frames, mel.shape[3]), float(mel.min()))
        mel = torch.cat([mel, pad], dim=2)
    elif frames > target_frames:
        mel = mel[:, :, :target_frames, :]
    return ldm.vae_encode(mel).cpu()


@torch.no_grad()
def real_pairs(
    ldm, x0: torch.Tensor, cond: dict, seed: int
) -> dict[str, Any]:
    """Forward-noise a real latent to every DDIM timestep and form exact-DDIM-step pairs.

    For each grid timestep t (noisier): draw an independent Gaussian, form the forward-diffusion
    state x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, take the teacher's epsilon there, then one
    exact DDIM reverse step to the cleaner latent x_clean. The training pair is
    (x_clean, timestep = t) -> teacher epsilon at x_t -- the shifted-timestep substitution
    AudioLDM2 DDIM inversion makes, so a perfect student makes that inversion step exact. The
    noisier states are kept in `states` for the step-invariant verifier.

    Args:
        ldm: `AudioLDM2Wrapper`.
        x0: Clean VAE latent `[1, C, H, W]` on the device.
        cond: `{encoder_hidden_states, class_labels, encoder_attention_mask}` on the device.
        seed: Seed for the per-timestep noise draws.

    Returns:
        Dict with trajectory `[N+1, ...]` (row i+1 is pair i's input), states `[N, ...]`,
        target_eps `[N, ...]`, and the timestep grid.
    """
    scheduler = ldm.model.scheduler
    timesteps = scheduler.timesteps
    abar = scheduler.alphas_cumprod.to(x0.device)
    gen = torch.Generator(device=x0.device).manual_seed(seed)

    states, xclean, targets, ts = [], [], [], []
    for t in timesteps:
        a = abar[int(t)]
        eps = torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
        eps_hat = ldm.unet_forward(
            scheduler.scale_model_input(x_t, t), timestep=t, **cond
        )[0].sample
        x_clean = scheduler.step(eps_hat, t, x_t, eta=0).prev_sample
        states.append(x_t.cpu())
        xclean.append(x_clean.cpu())
        targets.append(eps_hat.cpu())
        ts.append(int(t.item()) if hasattr(t, "item") else int(t))

    states = torch.cat(states)
    trajectory = torch.cat([states[:1]] + [x for x in xclean])  # row 0 unused as x_clean
    return {"trajectory": trajectory, "states": states,
            "target_eps": torch.cat(targets), "timesteps": ts}


def save_sample(sample_dir: Path, sample: dict, record: dict, cond: dict, meta: dict,
                store_dtype: torch.dtype) -> None:
    """Write one real-pairs sample in the layout `AudioLDM2TrajectoryDataset` reads, plus states.

    `cond` carries the per-sample conditioning under the dataset's own keys
    (`generated_prompt_embeds`, `t5_prompt_embeds`, `t5_attention_mask`).
    """
    (sample_dir / "latents").mkdir(parents=True, exist_ok=True)
    (sample_dir / "targets").mkdir(parents=True, exist_ok=True)
    traj = sample["trajectory"].to(store_dtype)
    tgt = sample["target_eps"].to(store_dtype)
    assert traj.shape[0] == tgt.shape[0] + 1, (traj.shape, tgt.shape)
    assert len(sample["timesteps"]) == tgt.shape[0], (len(sample["timesteps"]), tgt.shape)
    torch.save(traj, sample_dir / "latents/trajectory.pt")
    torch.save(sample["states"].to(store_dtype), sample_dir / "latents/states.pt")
    torch.save(tgt, sample_dir / "targets/target_eps.pt")
    torch.save({k: v.cpu() for k, v in cond.items()}, sample_dir / "conditioning.pt")
    with (sample_dir / "timesteps.json").open("w", encoding="utf-8") as f:
        json.dump(sample["timesteps"], f)
    with (sample_dir / "prompt.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    meta = {**meta, "trajectory_length": int(traj.shape[0]), "num_transitions": int(tgt.shape[0]),
            "latent_shape": list(traj.shape[1:]),
            "t5_seq_len": int(cond["t5_prompt_embeds"].shape[0])}
    tmp = sample_dir / "meta.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tmp.replace(sample_dir / "meta.json")


def load_clip_records(csv_path: Path, audio_dir: Path, num_captions: int) -> list[dict]:
    """Caption rows whose MusicCaps clip is on disk, in caption-file order."""
    meta = pd.read_csv(csv_path).head(num_captions)
    records = []
    for _, row in meta.iterrows():
        wav = audio_dir / f"[{row.ytid}]-[{int(row.start_s)}-{int(row.end_s)}].wav"
        if wav.exists():
            records.append({"prompt": str(row.caption), "ytid": str(row.ytid), "wav": str(wav)})
    return records


@hydra.main(config_path="../../config", config_name="generate_real_pairs_audioldm2",
            version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate the AudioLDM2 real-audio forward-noise training dataset."""
    from dotenv import load_dotenv

    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.device))
    if device.type == "cpu":
        logger.warning("Running on CPU; use only for smoke tests.")
    else:
        torch.cuda.set_device(device)

    records = load_clip_records(Path(cfg.captions_csv), Path(cfg.audio_dir), int(cfg.num_captions))
    if not records:
        raise FileNotFoundError(f"no clips from {cfg.captions_csv} under {cfg.audio_dir}")
    draws = int(cfg.draws_per_clip)
    total = len(records) * draws
    start = int(cfg.start_index)
    end = total if cfg.num_samples is None else start + int(cfg.num_samples)
    logger.info("{} clips x {} draws = {} samples; shard [{}, {})",
                len(records), draws, total, start, min(end, total))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    logger.info("Loading {} ({} steps)", cfg.model_id, cfg.num_inference_steps)
    ldm = load_model(str(cfg.model_id), device, int(cfg.num_inference_steps), edit_method="ddim")
    ldm.model.unet.eval()
    # latent_height already returns the mel frame count the VAE consumes (1024 for 10.24 s); the
    # VAE downsamples that by vae_scale_factor to the [8, 256, 16] latent the trajectory dataset
    # uses. Do NOT multiply again -- that inflates the clip to ~40 s and the latent to [8,1024,16].
    target_frames = latent_height(ldm.model, cfg.audio_length_in_s)
    logger.info("scheduler={} prediction_type={} steps={} target_mel_frames={}",
                type(ldm.model.scheduler).__name__, ldm.model.scheduler.config.prediction_type,
                len(ldm.model.scheduler.timesteps), target_frames)

    store_dtype = getattr(torch, str(cfg.store_dtype))
    meta_base = {"model_id": str(cfg.model_id), "guidance_scale": 1.0,
                 "num_inference_steps": int(cfg.num_inference_steps),
                 "audio_length_in_s": cfg.audio_length_in_s, "pairing": "shifted_timestep",
                 "target_space": "epsilon", "input_space": "exact_ddim_step",
                 "states_source": "real_audio_forward_noise", "store_dtype": str(cfg.store_dtype),
                 "git_sha": git_sha()}

    encoded: dict[int, Any] = {}
    first = True
    for sample_idx in tqdm(range(start, min(end, total)), desc="samples"):
        sample_dir = out_dir / f"sample_{sample_idx:06d}"
        if (sample_dir / "meta.json").exists() and not cfg.overwrite:
            continue
        clip_idx, draw = divmod(sample_idx, draws)
        record = records[clip_idx]
        if clip_idx not in encoded:
            x0 = encode_clip(ldm, Path(record["wav"]), target_frames).to(device)
            hidden, t5, mask = ldm.encode_text([record["prompt"]])
            # For the unet forward; the dataset's own key names are set at save time below.
            cond = {"encoder_hidden_states": hidden, "class_labels": t5,
                    "encoder_attention_mask": mask}
            save_cond = {"generated_prompt_embeds": hidden[0].cpu(),
                         "t5_prompt_embeds": t5[0].cpu(),
                         "t5_attention_mask": mask[0].cpu()}
            encoded = {clip_idx: (x0, cond, save_cond)}
        x0, cond, save_cond = encoded[clip_idx]

        seed = int(cfg.seed_base) + sample_idx
        sample = real_pairs(ldm, x0, cond, seed)
        if first:
            # Each input (trajectory[i+1]) is one DDIM reverse step from its forward-noised state
            # (states[i]); both arrays are length N, so compare them directly.
            gap = (sample["trajectory"][1:] - sample["states"]).flatten(1).norm(dim=1) / \
                sample["states"].flatten(1).norm(dim=1)
            logger.info("First clip {!r}: latent {} noisiest-state RMS {:.3g} step(input-vs-state) "
                        "rel mean {:.3e}", record["prompt"][:70], tuple(x0.shape[1:]),
                        float(sample["states"][0].pow(2).mean().sqrt()), float(gap.mean()))
            first = False

        save_sample(sample_dir, sample, {"prompt": record["prompt"], "ytid": record["ytid"],
                                         "wav": record["wav"]}, save_cond,
                    {**meta_base, "sample_idx": sample_idx, "seed": seed, "clip_idx": clip_idx,
                     "draw": draw}, store_dtype)

    logger.success("shard [{}, {}) done under {}", start, min(end, total), out_dir)


if __name__ == "__main__":
    main()
