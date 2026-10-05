# ABOUTME: Noise-normality vs reconstruction benchmark: generate clips from known noise, fully invert
# ABOUTME: and reconstruct them with DDIM, DDIM+LoRA and DDPM, plus a fresh-Gaussian-noise baseline.

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_inversion.inversion_utils import (  # noqa: E402
    inversion_forward_process,
    inversion_reverse_process,
)
from models import load_model  # noqa: E402

from src.inversion_lora.apply_lora import attach_inversion_lora  # noqa: E402
from src.inversion_lora.generate_trajectories import git_sha, latent_height  # noqa: E402
from src.inversion_lora.reconstruct import ddim_denoise, ddim_invert, encode_prompts  # noqa: E402
from src.inversion_lora.stable_audio import (  # noqa: E402
    ExactDPMSolver,
    StableAudioTeacher,
    decode_to_audio,
    ode_denoise,
    ode_invert,
)

ARMS = ("ddim", "ddim_lora", "ddpm", "gaussian")
# The DDPM-inversion settings the editing benchmark runs with (audioldm_run / stable_audio_run).
DDPM_ETA = 1.0


def benchmark_prompts(prompts_csv: str | Path, num_prompts: int, seed: int) -> list[str]:
    """Draw the benchmark captions from the MedleyMD source and target captions.

    These never entered LoRA training, which used MusicCaps only. Sorting the unique captions
    before drawing makes the draw independent of CSV row order.

    Args:
        prompts_csv: MedleyMD prompt CSV with `source_captions` and `target_captions`.
        num_prompts: How many distinct captions to draw.
        seed: Seed of the draw.

    Returns:
        `num_prompts` distinct captions.
    """
    frame = pd.read_csv(prompts_csv)
    pool = sorted(set(frame["source_captions"]) | set(frame["target_captions"]))
    assert len(pool) >= num_prompts, (len(pool), num_prompts)
    picked = np.random.default_rng(seed).choice(len(pool), size=num_prompts, replace=False)
    return [str(pool[i]) for i in picked]


def sample_table(cfg: DictConfig) -> pd.DataFrame:
    """One row per benchmark sample: its caption and the three seeds it uses.

    Generation, the Gaussian baseline and DDPM's per-step noise draw from disjoint seed ranges, so
    no arm ever reuses the noise a sample was generated from.

    Args:
        cfg: The run config.

    Returns:
        Frame indexed by sample, with `prompt`, `seed_gen`, `seed_gaussian` and `seed_ddpm`.
    """
    prompts = benchmark_prompts(
        AUDIO_ROOT / cfg.prompts_csv, int(cfg.num_prompts), int(cfg.prompt_seed)
    )
    count = len(prompts) * int(cfg.seeds_per_prompt)
    index = np.arange(count)
    seeds = {
        f"seed_{name}": int(cfg[f"seed_{name}"]) + index for name in ("gen", "gaussian", "ddpm")
    }
    ranges = [set(values.tolist()) for values in seeds.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(ranges) for b in ranges[i + 1 :]), (
        "seed ranges overlap: an arm would reuse another arm's noise"
    )
    return pd.DataFrame(
        {"prompt": [prompts[i // int(cfg.seeds_per_prompt)] for i in index], **seeds}, index=index
    )


class AudioLDM2Backend:
    """AudioLDM2 on its 100-step DDIM grid: batched DDIM passes, the editing code's DDPM pass."""

    sample_rate = 16000

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.steps = int(cfg.num_inference_steps)
        self.ldm = load_model(str(cfg.model_id), device, self.steps, edit_method="ddim")
        # The latent shape `AudioLDM2Pipeline.prepare_latents` builds for this duration.
        pipe = self.ldm.model
        self.shape = (
            pipe.unet.config.in_channels,
            latent_height(pipe, float(cfg.duration_s)) // pipe.vae_scale_factor,
            int(pipe.vocoder.config.model_in_dim) // pipe.vae_scale_factor,
        )
        self.noise_scale = float(self.ldm.model.scheduler.init_noise_sigma)
        self.denoiser = self.ldm.model.unet

    def noise(self, seed: int) -> torch.Tensor:
        """Initial latent `[1, C, H, W]` for one seed, drawn on the CPU so it is portable."""
        generator = torch.Generator().manual_seed(seed)
        return torch.randn((1, *self.shape), generator=generator) * self.noise_scale

    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Full deterministic DDIM sampling from the noisiest timestep."""
        return ddim_denoise(self.ldm, x_t.to(self.device), encode_prompts(self.ldm, prompts)).cpu()

    @torch.no_grad()
    def invert(self, x0: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Full DDIM inversion to the noisiest timestep."""
        return ddim_invert(self.ldm, x0.to(self.device), encode_prompts(self.ldm, prompts)).cpu()

    def ddpm(self, x0: torch.Tensor, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Edit-friendly DDPM inversion and reconstruction of one latent, via the editing code.

        Returns:
            The sampled noisiest latent x_T and the reconstruction, both `[1, ...]` on the CPU.
        """
        return run_ddpm(self.ldm, x0.to(self.device), prompt, seed, self.cfg, duration=None)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Waveform `[1, samples]` of one latent, through the VAE and the HiFi-GAN vocoder."""
        mel = self.ldm.vae_decode(latent.to(self.device))
        return self.ldm.decode_to_mel(mel).detach().cpu().float()


class StableAudioBackend:
    """Stable Audio Open on its native cosine grid.

    The deterministic passes run the first-order ODE with its exact inverse (the editing code's
    `odeinv`), batched through `StableAudioTeacher`. DDPM inversion runs through the editing
    code's `StableAudWrapper`. Both views share one pipeline, so there is one set of weights.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.steps = int(cfg.num_inference_steps)
        self.duration_s = float(cfg.duration_s)
        self.wrapper = load_model(
            str(cfg.model_id), device, self.steps, token=os.getenv("HF_TOKEN"), edit_method="ddpm"
        )
        self.teacher = StableAudioTeacher(self.wrapper.model, device, self.duration_s, self.steps)
        self.solver = ExactDPMSolver(self.wrapper.model.scheduler)
        self.sample_rate = int(self.wrapper.model.vae.config.sampling_rate)
        self.shape = (
            self.wrapper.model.transformer.config.in_channels,
            self.teacher.latent_length,
        )
        # The ODE starts at sigma_max: x_T = sigma_max * eps. Dividing by it puts every arm's
        # noise in unit-variance units; the KL and correlation are invariant to the common scale.
        self.noise_scale = float(self.solver.sigmas[0])
        self.denoiser = self.wrapper.model.transformer
        self._check_conditioning_matches_wrapper()

    def _check_conditioning_matches_wrapper(self) -> None:
        """Assert the teacher and the editing wrapper see the same conditioning.

        The deterministic arms run through the teacher, DDPM through the wrapper; one forward on
        the same input must agree, or the two families would be scored on different models.
        """
        prompt = "A test caption."
        x = torch.randn((1, *self.shape), generator=torch.Generator().manual_seed(0)).to(
            self.device
        )
        index = self.steps // 2
        timestep = torch.tensor(self.solver.timesteps[index], device=self.device)
        with torch.no_grad():
            ours = self.teacher.forward(
                self.solver.model_input(x, index),
                timestep.reshape(1),
                self.teacher.encode_prompt(prompt),
            )
            self.wrapper.setup_extra_inputs(
                x, init_timestep=timestep, audio_end_in_s=self.duration_s
            )
            embeds, _, mask = self.wrapper.encode_text([prompt])
            theirs = self.wrapper.unet_forward(
                self.solver.model_input(x, index),
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_attention_mask=mask,
            )[0].sample
        gap = float((ours - theirs).abs().max() / theirs.abs().max())
        logger.info("teacher vs editing-wrapper forward: max relative gap {:.2e}", gap)
        assert gap < 1e-4, f"teacher and wrapper disagree by {gap:.2e}"

    def noise(self, seed: int) -> torch.Tensor:
        """Initial latent `[1, C, L]` at sigma_max for one seed, drawn on the CPU."""
        generator = torch.Generator().manual_seed(seed)
        return torch.randn((1, *self.shape), generator=generator) * self.noise_scale

    def _predict(self, prompts: list[str]):
        """Batched `predict(x, index) -> data prediction` under the prompts' conditioning."""
        text_audio = torch.cat([self.teacher.encode_prompt(p) for p in prompts])

        def predict(x: torch.Tensor, index: int) -> torch.Tensor:
            t = torch.full((x.shape[0],), self.solver.timesteps[index], device=self.device)
            raw = self.teacher.forward(self.solver.model_input(x, index), t, text_audio)
            return self.solver.data_prediction(x, raw, index)

        return predict

    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Full ODE sampling from sigma_max, including the final step to sigma = 0."""
        return ode_denoise(self.solver, x_t.to(self.device), 0, self._predict(prompts)).cpu()

    @torch.no_grad()
    def invert(self, x0: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Full exact-inverse ODE inversion; the final sigma = 0 step has no inverse."""
        return ode_invert(
            self.solver, x0.to(self.device), self._predict(prompts), self.solver.invertible_steps
        ).cpu()

    def ddpm(self, x0: torch.Tensor, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Edit-friendly DDPM inversion and reconstruction of one latent, via the editing code."""
        return run_ddpm(self.wrapper, x0.to(self.device), prompt, seed, self.cfg, self.duration_s)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Stereo waveform `[2, samples]` of one latent, cropped to the conditioned duration."""
        return decode_to_audio(self.teacher, latent.to(self.device))[0]


@torch.no_grad()
def run_ddpm(
    model, x0: torch.Tensor, prompt: str, seed: int, cfg: DictConfig, duration: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full DDPM inversion then reconstruction, called exactly as the editing code calls it.

    The forward process draws one independent noise per timestep with the global RNG, so it is
    seeded here from the sample's own DDPM seed. Re-gridding first clears the multistep solver's
    step index and output history, which would otherwise leak from the previous sample.

    Args:
        model: Editing-code `PipelineWrapper`.
        x0: Clean latent `[1, ...]` on the model's device.
        prompt: Caption for both passes.
        seed: Seed for the forward process's noise draws.
        cfg: Run config, for the step count and guidance.
        duration: Audio duration for Stable Audio's timing conditioning; None for AudioLDM2.

    Returns:
        The noisiest latent x_T and the reconstruction, both `[1, ...]` on the CPU.
    """
    steps = int(cfg.num_inference_steps)
    guidance = float(cfg.guidance_scale)
    model.model.scheduler.set_timesteps(steps, device=model.device)
    torch.manual_seed(seed)
    _, zs, xts, extra_info = inversion_forward_process(
        model,
        x0,
        etas=DDPM_ETA,
        prompts=[prompt],
        cfg_scales=[guidance],
        num_inference_steps=steps,
        numerical_fix=True,
        duration=duration,
    )
    assert xts.shape[0] == steps + 1, xts.shape
    x_t = xts[steps][None].clone()
    reconstruction, _ = inversion_reverse_process(
        model,
        xT=xts,
        tstart=torch.tensor([steps], dtype=torch.int),
        etas=DDPM_ETA,
        prompts=[prompt],
        neg_prompts=[""],
        cfg_scales=[guidance],
        zs=zs[:steps],
        duration=duration,
        extra_info=extra_info,
    )
    return x_t.cpu(), reconstruction.cpu()


def save_wavs(backend, latents: torch.Tensor, indices: list[int], out_dir: Path) -> None:
    """Decode each latent and write `<index>.wav` as float32, unclipped.

    Float32 rather than 16-bit PCM, so quantisation neither floors the reconstruction metrics nor
    rounds a near-identical reconstruction into a bit-identical file with infinite mel PSNR.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for latent, index in zip(latents, indices):
        audio = backend.decode(latent[None])
        assert audio.ndim == 2 and audio.shape[1] > 0, audio.shape
        # Explicit: torchaudio's default for a float tensor here is 16-bit PCM.
        torchaudio.save(
            str(out_dir / f"{index:04d}.wav"),
            audio,
            sample_rate=backend.sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )


def batched(
    fn, latents: torch.Tensor, prompts: list[str], batch_size: int, label: str
) -> torch.Tensor:
    """Apply `fn(latents, prompts)` batch by batch, logging progress and ETA."""
    out = []
    start = time.time()
    for begin in range(0, latents.shape[0], batch_size):
        end = min(begin + batch_size, latents.shape[0])
        out.append(fn(latents[begin:end], prompts[begin:end]))
        elapsed = time.time() - start
        logger.info(
            "{} {}/{} ({:.0f}s, ETA {:.0f}s)",
            label,
            end,
            latents.shape[0],
            elapsed,
            elapsed / end * (latents.shape[0] - end),
        )
    return torch.cat(out)


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of `a` against reference `b`, over the whole batch."""
    return float((a - b).norm() / b.norm())


@hydra.main(config_path="../../config", config_name="noise_recon_audioldm2", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run every arm on one shard of the benchmark and save its latents and wavs."""
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    started = time.time()
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    run_dir = AUDIO_ROOT / str(cfg.run_dir)
    shard_path = run_dir / "latents" / f"shard{int(cfg.shard_id):02d}.pt"
    if shard_path.exists() and not cfg.overwrite:
        logger.info("{} exists; nothing to do (overwrite=true to redo)", shard_path)
        return
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        run_dir / "logs" / f"shard{int(cfg.shard_id):02d}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logger.add(str(log_path))

    table = sample_table(cfg)
    shards = np.array_split(table.index.to_numpy(), int(cfg.num_shards))
    shard = table.loc[shards[int(cfg.shard_id)]]
    if cfg.limit is not None:
        shard = shard.head(int(cfg.limit))
    indices = shard.index.tolist()
    prompts = shard["prompt"].tolist()
    logger.info(
        "shard {}/{}: samples {}..{} ({} of {}), first prompt {!r}",
        int(cfg.shard_id),
        int(cfg.num_shards),
        indices[0],
        indices[-1],
        len(indices),
        len(table),
        prompts[0],
    )

    device = torch.device(str(cfg.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backend = {"audioldm2": AudioLDM2Backend, "stable_audio": StableAudioBackend}[str(cfg.model)](
        cfg, device
    )
    batch = int(cfg.batch_size)
    wav_root = run_dir / "wavs"

    noise = torch.cat([backend.noise(int(s)) for s in shard["seed_gen"]])
    logger.info("noise {} {} std {:.4f}", tuple(noise.shape), noise.dtype, float(noise.std()))
    x0 = batched(backend.denoise, noise, prompts, batch, "generate")
    logger.info("x0 {} mean {:.4f} std {:.4f}", tuple(x0.shape), float(x0.mean()), float(x0.std()))
    save_wavs(backend, x0, indices, wav_root / "source")

    noises = {}

    def finish(arm: str, x_t: torch.Tensor, reconstruction: torch.Tensor) -> None:
        """Keep the arm's noise, log its round-trip errors, write its reconstructions."""
        noises[arm] = x_t
        logger.info(
            "{}: x_T std {:.4f}, rel. error to generation noise {:.4f}, latent rel. error {:.5f}",
            arm,
            float(x_t.std() / backend.noise_scale),
            relative_error(x_t, noise),
            relative_error(reconstruction, x0),
        )
        save_wavs(backend, reconstruction, indices, wav_root / arm)

    gaussian = torch.cat([backend.noise(int(s)) for s in shard["seed_gaussian"]])
    finish("gaussian", gaussian, batched(backend.denoise, gaussian, prompts, batch, "gaussian"))

    x_t = batched(backend.invert, x0, prompts, batch, "ddim invert")
    finish("ddim", x_t, batched(backend.denoise, x_t, prompts, batch, "ddim denoise"))

    ddpm_noise, ddpm_rec = [], []
    ddpm_start = time.time()
    for pos, row in enumerate(shard.itertuples()):
        x_t, rec = backend.ddpm(x0[pos : pos + 1], str(row.prompt), int(row.seed_ddpm))
        ddpm_noise.append(x_t)
        ddpm_rec.append(rec)
        elapsed = time.time() - ddpm_start
        logger.info(
            "ddpm {}/{} ({:.0f}s, ETA {:.0f}s)",
            pos + 1,
            len(shard),
            elapsed,
            elapsed / (pos + 1) * (len(shard) - pos - 1),
        )
    finish("ddpm", torch.cat(ddpm_noise), torch.cat(ddpm_rec))

    # Last, because injecting the adapter perturbs the base forward even while disabled (cuBLAS
    # rounding through the wrapped Linears), and every other arm must run the untouched model.
    # It is merged once for the whole shard and applies to the inversion pass only.
    set_enabled = attach_inversion_lora(backend.denoiser, str(cfg.lora_path))
    set_enabled(True)
    x_t = batched(backend.invert, x0, prompts, batch, "ddim_lora invert")
    set_enabled(False)
    finish("ddim_lora", x_t, batched(backend.denoise, x_t, prompts, batch, "ddim_lora denoise"))

    assert set(noises) == set(ARMS), sorted(noises)
    torch.save(
        {
            "indices": indices,
            "prompts": prompts,
            "seeds": shard[["seed_gen", "seed_gaussian", "seed_ddpm"]].to_dict("list"),
            "reference": noise / backend.noise_scale,
            "x0": x0,
            "noise": {arm: value / backend.noise_scale for arm, value in noises.items()},
            "noise_scale": backend.noise_scale,
        },
        shard_path.with_suffix(".tmp"),
    )
    shard_path.with_suffix(".tmp").rename(shard_path)

    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "host": platform.node(),
        "samples": len(indices),
        "started": datetime.fromtimestamp(started).isoformat(),
        "wall_clock_s": time.time() - started,
    }
    (run_dir / f"run_meta_shard{int(cfg.shard_id):02d}.json").write_text(
        json.dumps(meta, indent=2)
    )
    logger.success(
        "shard {} done in {:.0f}s -> {}", int(cfg.shard_id), meta["wall_clock_s"], shard_path
    )


if __name__ == "__main__":
    main()
