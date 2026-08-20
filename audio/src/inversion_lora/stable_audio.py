# ABOUTME: Stable Audio Open teacher for the inversion LoRA: conditioning, batched DiT forwards
# ABOUTME: with per-example timesteps, and deterministic reverse trajectories to distil from.

from types import SimpleNamespace

import torch
from diffusers import (
    CosineDPMSolverMultistepScheduler,
    DPMSolverMultistepScheduler,
    StableAudioPipeline,
)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-audio-open-1.0"

# Stable Audio's native scheduler is an SDE: it draws Brownian noise per step from an unseeded
# sampler, so its trajectories are irreproducible and DDIM-style inversion is undefined on it.
# `beta` is the deterministic replacement the editing code's `ddim` mode builds, and the teacher
# the adapter is trained against. It runs the DiT on a linear-beta grid (sigma 157..0.05,
# t 999..10) rather than the cosine grid the model was trained on (sigma 500..0.3, t 0.99..0.19);
# see output/sao_probe/REPORT.md. `cosine` is kept so the probe can compare the two.
SCHEDULES = ("beta", "cosine")


class StableAudioTeacher:
    """A `StableAudioPipeline` viewed as the trainer's frozen teacher plus student denoiser.

    Exposes the `PipelineWrapper` surface the LoRA trainer needs (`model.unet`,
    `model.scheduler`, `device`) so the AudioLDM2 training loop applies unchanged, and holds the
    conditioning that is constant across a run: the timing embeddings and the rotary positional
    embedding, both of which depend only on the duration.
    """

    def __init__(
        self,
        pipe: StableAudioPipeline,
        device: torch.device,
        duration_s: float | None,
        num_inference_steps: int,
    ):
        """Wrap a loaded pipeline and precompute its duration-dependent conditioning.

        Args:
            pipe: Pipeline with its scheduler and timesteps already set.
            device: Device the pipeline is on.
            duration_s: Duration the timing conditioning declares; None uses the model's maximum.
            num_inference_steps: Length of the sampling grid, re-applied before each trajectory.
        """
        self.pipe = pipe
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.duration_s = self.max_duration_s if duration_s is None else float(duration_s)
        assert 0 < self.duration_s <= self.max_duration_s, (
            f"duration_s={self.duration_s} outside (0, {self.max_duration_s}]"
        )
        # The trainer addresses the denoiser as `ldm.model.unet` and freezes `model.<component>`.
        self.model = SimpleNamespace(
            unet=pipe.transformer,
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            projection_model=pipe.projection_model,
        )

        # Under no_grad: this conditioning is constant for the whole run and reused by every
        # forward, so a graph through the frozen projection model would be freed by the first
        # backward and break the second.
        with torch.no_grad():
            seconds_start, seconds_end = pipe.encode_duration(
                0.0, self.duration_s, device, False, 1
            )
        self.seconds_hidden_states = (seconds_start, seconds_end)
        self.global_states = torch.cat([seconds_start, seconds_end], dim=2)
        self.rotary = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            self.latent_length + self.global_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )

    @property
    def max_duration_s(self) -> float:
        """Longest audio the model conditions on, from its fixed latent length."""
        return (
            self.latent_length
            * self.pipe.vae.hop_length
            / self.pipe.vae.config.sampling_rate
        )

    @property
    def latent_length(self) -> int:
        """The latent time dimension, which is fixed regardless of the requested duration."""
        return int(self.pipe.transformer.config.sample_size)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode one caption into the DiT's cross-attention states, timing embeddings included.

        The timing embeddings are appended here rather than at training time so a cached
        trajectory carries everything the forward needs in one tensor.

        Args:
            prompt: Caption to condition on.

        Returns:
            Cross-attention states `[1, tokens + 2, dim]`.
        """
        prompt_embeds = self.pipe.encode_prompt([prompt], self.device, False)
        return torch.cat([prompt_embeds, *self.seconds_hidden_states], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_audio: torch.Tensor) -> torch.Tensor:
        """Run the DiT on a batch of latents with one timestep per element.

        The pipeline passes a single scalar timestep for the whole batch; training needs one per
        element, which the DiT supports directly because the time embedding is broadcast over the
        sequence. Conditioning with a batch of 1 is expanded, so a single prompt can drive a whole
        batch of transitions.

        Args:
            x: Latents `[B, C, L]`.
            t: Timesteps `[B]`, in the scheduler's own units.
            text_audio: Cross-attention states `[B, S, D]` or `[1, S, D]`.

        Returns:
            The DiT's prediction, same shape as `x`.
        """
        batch = x.shape[0]
        assert x.ndim == 3, x.shape
        assert t.shape == (batch,), (t.shape, x.shape)
        assert text_audio.shape[0] in (1, batch), (text_audio.shape, batch)
        out = self.pipe.transformer(
            x,
            t,
            encoder_hidden_states=text_audio.expand(batch, -1, -1),
            global_hidden_states=self.global_states.expand(batch, -1, -1),
            rotary_embedding=self.rotary,
            return_dict=False,
        )[0]
        assert out.shape == x.shape, (out.shape, x.shape)
        return out

    @torch.no_grad()
    def reverse_trajectory(
        self, text_audio: torch.Tensor, seed: int, progress: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Sample without guidance, keeping every latent and every teacher prediction.

        No CFG, so the trajectory is self-consistent with the no-CFG distillation loss.

        Args:
            text_audio: Cross-attention states `[1, S, D]` from `encode_prompt`.
            seed: Seed for the initial latent.
            progress: Show a per-step progress bar.

        Returns:
            `(trajectory, outputs, timesteps)`, where `trajectory[i]` is noisier than
            `trajectory[i + 1]` and `outputs[i]` is the teacher's prediction at
            `(trajectory[i], timesteps[i])`, all on the CPU.
        """
        scheduler = self.pipe.scheduler
        # Multistep solvers carry state across a trajectory (step index, the previous model
        # outputs, the current solver order). Re-gridding resets all of it, so trajectory N + 1
        # neither inherits trajectory N's history nor runs its step index off the end.
        scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        shape = (1, self.pipe.transformer.config.in_channels, self.latent_length)
        latents = (
            torch.randn(
                shape,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                device=self.device,
            )
            * scheduler.init_noise_sigma
        )

        trajectory = [latents.cpu()]
        outputs: list[torch.Tensor] = []
        timesteps: list[float] = []
        for t in tqdm(scheduler.timesteps, desc="denoising", leave=False, disable=not progress):
            model_input = scheduler.scale_model_input(latents, t)
            prediction = self.forward(model_input, t.reshape(1).to(self.device), text_audio)
            outputs.append(prediction.cpu())
            latents = scheduler.step(prediction, t, latents).prev_sample
            trajectory.append(latents.cpu())
            timesteps.append(float(t))

        trajectory = torch.cat(trajectory)
        outputs = torch.cat(outputs)
        assert trajectory.shape[0] == outputs.shape[0] + 1, (trajectory.shape, outputs.shape)
        assert len(timesteps) == outputs.shape[0], (len(timesteps), outputs.shape)
        return trajectory, outputs, timesteps


def load_teacher(
    model_id: str,
    device: torch.device,
    num_inference_steps: int,
    duration_s: float | None = None,
    schedule: str = "beta",
) -> StableAudioTeacher:
    """Load Stable Audio Open on a deterministic sampling grid.

    Args:
        model_id: Hub id of the pipeline.
        device: Device to place the pipeline on.
        num_inference_steps: Length of the sampling grid; the editing baselines use 100.
        duration_s: Duration the timing conditioning declares; None uses the model's maximum.
        schedule: `beta` for the editing code's DDIM scheduler, `cosine` for the native SDE.

    Returns:
        The teacher, in eval mode with its timesteps set.
    """
    assert schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}, got {schedule!r}"
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    assert isinstance(pipe.scheduler, CosineDPMSolverMultistepScheduler), type(pipe.scheduler)
    if schedule == "beta":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    return StableAudioTeacher(pipe, device, duration_s, num_inference_steps)


@torch.no_grad()
def decode_to_audio(teacher: StableAudioTeacher, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to waveform and crop to the conditioned duration.

    Args:
        teacher: The loaded teacher.
        latents: Latents `[B, C, L]`.

    Returns:
        Waveform `[B, channels, samples]` on the CPU.
    """
    audio = teacher.pipe.vae.decode(latents.to(teacher.device)).sample
    samples = int(teacher.duration_s * teacher.pipe.vae.config.sampling_rate)
    return audio[:, :, :samples].detach().cpu().float()
