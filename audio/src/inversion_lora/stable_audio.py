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
from loguru import logger
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-audio-open-1.0"

# `cosine` is Stable Audio's own sigma grid (500..0.3, t 0.99..0.19) and the only one used. Its
# scheduler is an SDE -- unseeded Brownian noise per step -- so trajectories come from
# `FirstOrderSolver` instead of `scheduler.step`, which is deterministic and exactly invertible.
#
# `beta` is what the editing code's `ddim` mode builds by rebuilding the scheduler as
# DPMSolverMultistepScheduler, which silently drops sigma_min/sigma_max and lands on a linear-beta
# grid (sigma 157..0.05, t 999..10). REJECTED: the DiT is then queried ~1000x outside its trained
# timestep range, its data predictions come out ~100x too large and the decoded audio clips by 22x
# (output/sao_schedules/REPORT.md). It is kept only so that comparison stays reproducible.
SCHEDULES = ("cosine", "beta")


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
    def ode_trajectory(
        self, text_audio: torch.Tensor, seed: int, progress: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Sample deterministically with the first-order ODE, keeping latents and data predictions.

        This is the trajectory the inversion LoRA is trained on. Unlike `reverse_trajectory` it does
        not call `scheduler.step`, so it is reproducible on the cosine grid (whose scheduler is an
        SDE) and exactly invertible by `FirstOrderSolver.inverse`.

        Args:
            text_audio: Cross-attention states `[1, S, D]`.
            seed: Seed for the initial latent.
            progress: Show a per-step progress bar.

        Returns:
            `(trajectory, data, timesteps)`, where `data[i]` is the data prediction at
            `(trajectory[i], timesteps[i])` and the reverse step from `trajectory[i]` to
            `trajectory[i + 1]` consumed exactly that. All on the CPU.
        """
        solver = FirstOrderSolver(self.model.scheduler)
        shape = (1, self.pipe.transformer.config.in_channels, self.latent_length)
        x = torch.randn(
            shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
        ) * solver.sigmas[0]

        trajectory, data = [x.cpu()], []
        for index in tqdm(
            range(len(solver.timesteps)), desc="sampling", leave=False, disable=not progress
        ):
            t = torch.tensor([solver.timesteps[index]], device=self.device)
            raw = self.forward(solver.model_input(x, index), t, text_audio)
            prediction = solver.data_prediction(x, raw, index)
            data.append(prediction.cpu())
            x = solver.forward(x, prediction, index)
            trajectory.append(x.cpu())
        return torch.cat(trajectory), torch.cat(data), list(solver.timesteps)

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
    schedule: str = "cosine",
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
    if schedule == "beta":
        logger.warning(
            "schedule=beta queries the DiT outside its trained timestep range: data predictions "
            "~100x too large, decoded audio clipping by 22x. See output/sao_schedules/REPORT.md. "
            "Use it only to reproduce that comparison."
        )
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


class FirstOrderSolver:
    """DPMSolver++ first order on a scheduler's sigma grid, with its exact algebraic inverse.

    The reverse step is affine in the sample -- `x_t = A * x_s + B * D`, with `D` the data
    prediction -- so recovering `x_s` from `x_t` is exact arithmetic given `D`. That is the property
    diffusers' `DPMSolverMultistepInverseScheduler` does not have: it builds its own grid, offset by
    one step from the reverse one, so no inverse step undoes any reverse step (see
    output/sao_pairing/REPORT.md). Everything here is stateless and indexed by step, so the
    inversion can walk the reverse grid backwards exactly.

    First order rather than the schedulers' default second order because the shifted-denoiser
    objective is only well posed per step: a multistep update mixes model outputs from several
    steps, so no single substitution defines its error.
    """

    def __init__(self, scheduler):
        """Read the grid, and detect whether the scheduler preconditions its inputs.

        Args:
            scheduler: A diffusers DPMSolver-family scheduler with `timesteps` already set.
        """
        self.scheduler = scheduler
        self.sigmas = scheduler.sigmas.clone()
        self.timesteps = [float(t) for t in scheduler.timesteps]
        # EDM-style schedulers scale the input by 1/sqrt(sigma^2 + sigma_data^2) and read the model
        # output through c_skip/c_out; VP ones feed the raw latent and convert v to x0 directly.
        self.preconditioned = hasattr(scheduler, "precondition_inputs")
        assert len(self.sigmas) == len(self.timesteps) + 1, (
            f"{len(self.sigmas)} sigmas for {len(self.timesteps)} timesteps"
        )

    @property
    def invertible_steps(self) -> int:
        """How many steps from the noisy end can be inverted.

        A schedule with `final_sigmas_type="zero"` ends with sigma_t = 0, where the reverse step is
        `x_t = 0 * x_s + B * D`: it discards the sample and returns the data prediction, so that one
        step destroys the information inversion needs. Both Stable Audio schedules do this, so the
        round trip runs from the second-cleanest latent instead.
        """
        return len(self.timesteps) - 1 if float(self.sigmas[-1]) == 0.0 else len(self.timesteps)

    def _alpha_sigma(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scheduler._sigma_to_alpha_sigma_t(self.sigmas[index])

    def model_input(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Scale the latent the way the scheduler's own `scale_model_input` would at `index`."""
        if not self.preconditioned:
            return x
        return self.scheduler.precondition_inputs(x, self.sigmas[index])

    def data_prediction(self, x: torch.Tensor, raw: torch.Tensor, index: int) -> torch.Tensor:
        """Convert the network's output at step `index` into a prediction of the clean latent.

        Args:
            x: The *unscaled* latent the network was run on.
            raw: The network's output.
            index: Step index into the grid.

        Returns:
            The data prediction `D`.
        """
        sigma = self.sigmas[index]
        if self.preconditioned:
            return self.scheduler.precondition_outputs(x, raw, sigma)
        alpha_t, sigma_t = self._alpha_sigma(index)
        assert self.scheduler.config.prediction_type == "v_prediction", (
            f"only v_prediction is wired here, got {self.scheduler.config.prediction_type}"
        )
        return alpha_t * x - sigma_t * raw

    def sigma_for(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map a batch of grid timesteps back to their sigmas, shaped for broadcasting.

        Args:
            timesteps: Timesteps `[B]`, each of which must be on this grid.

        Returns:
            Sigmas `[B, 1, 1]`.
        """
        grid = torch.tensor(self.timesteps, dtype=torch.float64)
        index = torch.searchsorted(grid.flip(0), timesteps.double().cpu().flip(0))
        index = (len(grid) - 1 - index.flip(0)).clamp(0, len(grid) - 1)
        chosen = grid[index]
        assert torch.allclose(chosen, timesteps.double().cpu(), atol=1e-6), (
            "timesteps are not on this solver's grid; the dataset and the solver disagree"
        )
        return self.sigmas[index].to(timesteps.device).reshape(-1, 1, 1)

    def model_input_batch(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """`model_input` for a batch with one sigma per element."""
        if not self.preconditioned:
            return x
        return self.scheduler.precondition_inputs(x, sigmas)

    def data_prediction_batch(
        self, x: torch.Tensor, raw: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        """`data_prediction` for a batch with one sigma per element."""
        if self.preconditioned:
            return self.scheduler.precondition_outputs(x, raw, sigmas)
        alpha_t = 1.0 / ((sigmas**2 + 1) ** 0.5)
        return alpha_t * x - sigmas * alpha_t * raw

    def coefficients(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The affine coefficients `(A, B)` of the reverse step from `index` to `index + 1`."""
        alpha_t, sigma_t = self._alpha_sigma(index + 1)
        alpha_s, sigma_s = self._alpha_sigma(index)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - (torch.log(alpha_s) - torch.log(sigma_s))
        return sigma_t / sigma_s, -(alpha_t * (torch.exp(-h) - 1.0))

    def forward(self, x_s: torch.Tensor, data: torch.Tensor, index: int) -> torch.Tensor:
        """One reverse step: noisier `x_s` at `index` to cleaner `x_t` at `index + 1`."""
        a, b = self.coefficients(index)
        return a * x_s + b * data

    def inverse(self, x_t: torch.Tensor, data: torch.Tensor, index: int) -> torch.Tensor:
        """Exact inverse of `forward`: recover the noisier latent at `index` from `x_t`."""
        a, b = self.coefficients(index)
        assert float(a) != 0.0, (
            f"step {index} maps every sample to the same point (sigma_t = 0); it has no inverse. "
            "Invert from `invertible_steps` instead."
        )
        return (x_t - b * data) / a
