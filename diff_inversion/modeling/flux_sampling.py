"""Sampling and explicit-Euler inversion helpers for FLUX flow matching.

Provenance:
- FLUX packing, transformer-call, and dynamic schedule conventions mirror Hugging Face
  Diffusers 0.37.1 ``FluxPipeline`` / ``FlowMatchEulerDiscreteScheduler`` (Apache-2.0):
  https://github.com/huggingface/diffusers/tree/v0.37.1/src/diffusers
- The reverse explicit-Euler equation follows Appendix P of *There and Back Again*:
  https://arxiv.org/abs/2410.23530v5
- Optional latent scaling exposes the controls used by Stable Flow and the UniEdit-Flow
  script bundled with the TABA code. No Stable Flow feature-injection code is used here.
- The optional fixed-point refinement follows the iterative-noising idea of ReNoise
  (arXiv:2403.14602), and the Heun predictor-corrector follows the algorithm exposed by
  Diffusers. Both reverse solvers are implemented locally for the exact FLUX schedule.

The shared schedule API, trajectory capture, and oracle inversion control are this
project's implementation. See ``docs/flow_matching_provenance.md`` for exact boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from PIL import Image
from tqdm import tqdm

FLUX_INVERSION_SOLVERS = frozenset({"euler", "fixed_point", "heun"})
FluxConditioning: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True)
class FluxSchedule:
    """The exact FLUX inference grid, including the terminal sigma."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor

    def __post_init__(self) -> None:
        validate_flux_schedule(self.timesteps, self.sigmas)

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])

    def cpu(self) -> FluxSchedule:
        return FluxSchedule(
            timesteps=self.timesteps.detach().cpu(),
            sigmas=self.sigmas.detach().cpu(),
        )


def validate_flux_schedule(timesteps: torch.Tensor, sigmas: torch.Tensor) -> None:
    if timesteps.ndim != 1 or sigmas.ndim != 1:
        raise ValueError(
            "FLUX timesteps and sigmas must be one-dimensional, got "
            f"{tuple(timesteps.shape)} and {tuple(sigmas.shape)}."
        )
    if sigmas.shape[0] != timesteps.shape[0] + 1:
        raise ValueError(
            "FLUX schedule must contain one terminal sigma more than timesteps: "
            f"got {sigmas.shape[0]} sigmas and {timesteps.shape[0]} timesteps."
        )
    if not bool(torch.isfinite(timesteps).all()) or not bool(torch.isfinite(sigmas).all()):
        raise ValueError("FLUX schedule contains non-finite values.")
    if bool((sigmas[1:] > sigmas[:-1]).any()):
        raise ValueError("FLUX sampling sigmas must be monotonically non-increasing.")


def _config_value(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def prepare_flux_schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    num_inference_steps: int,
    image_seq_len: int,
    device: torch.device | str,
) -> FluxSchedule:
    """Configure the scheduler exactly as ``FluxPipeline.__call__`` does."""
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}.")
    if image_seq_len <= 0:
        raise ValueError(f"image_seq_len must be positive, got {image_seq_len}.")

    scheduler_config = scheduler.config
    mu = calculate_shift(
        image_seq_len,
        _config_value(scheduler_config, "base_image_seq_len", 256),
        _config_value(scheduler_config, "max_image_seq_len", 4096),
        _config_value(scheduler_config, "base_shift", 0.5),
        _config_value(scheduler_config, "max_shift", 1.15),
    )
    sigmas = np.linspace(
        1.0,
        1.0 / num_inference_steps,
        num_inference_steps,
        dtype=np.float32,
    )
    if bool(_config_value(scheduler_config, "use_flow_sigmas", False)):
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    scheduler.set_begin_index(0)

    return FluxSchedule(
        timesteps=scheduler.timesteps.detach().clone(),
        sigmas=scheduler.sigmas.detach().clone(),
    )


def flow_euler_sampling_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """Apply one deterministic Euler sampling step from noisy to clean."""
    sigma_value = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    sigma_next_value = torch.as_tensor(sigma_next, device=sample.device, dtype=torch.float32)
    result = sample.float() + (sigma_next_value - sigma_value) * velocity.float()
    return result.to(dtype=velocity.dtype)


def flow_euler_inverse_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma_noisy: torch.Tensor | float,
    sigma_clean: torch.Tensor | float,
) -> torch.Tensor:
    """Apply one explicit ODE inversion step from clean to noisy."""
    sigma_noisy_value = torch.as_tensor(
        sigma_noisy,
        device=sample.device,
        dtype=torch.float32,
    )
    sigma_clean_value = torch.as_tensor(
        sigma_clean,
        device=sample.device,
        dtype=torch.float32,
    )
    result = sample.float() + (sigma_noisy_value - sigma_clean_value) * velocity.float()
    return result.to(dtype=velocity.dtype)


def validate_flux_inversion_solver(
    solver: str,
    fixed_point_refinement_steps: int,
) -> str:
    """Validate and normalize the reverse solver configuration."""
    solver = str(solver).strip().lower()
    if solver not in FLUX_INVERSION_SOLVERS:
        choices = ", ".join(sorted(FLUX_INVERSION_SOLVERS))
        raise ValueError(f"Unsupported FLUX inversion solver {solver!r}; choose one of: {choices}.")
    if int(fixed_point_refinement_steps) < 0:
        raise ValueError("fixed_point_refinement_steps must be non-negative.")
    if solver == "fixed_point" and int(fixed_point_refinement_steps) < 1:
        raise ValueError("fixed_point requires at least one refinement step.")
    return solver


def flux_inversion_nfe(
    num_steps: int,
    *,
    solver: str,
    fixed_point_refinement_steps: int = 1,
) -> int:
    """Return the deterministic number of transformer evaluations for inversion."""
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    solver = validate_flux_inversion_solver(solver, fixed_point_refinement_steps)
    evaluations_per_step = {
        "euler": 1,
        "fixed_point": 1 + int(fixed_point_refinement_steps),
        "heun": 2,
    }[solver]
    return int(num_steps) * evaluations_per_step


def nudge_flux_latent(latent: torch.Tensor, scalar: float) -> torch.Tensor:
    """Scale the clean latent before inversion for an explicitly requested control.

    Stable Flow uses 1.15 for a VAE-encoded real-image latent, while the reconstruction
    script bundled under TABA's UniEdit-Flow dependency uses 1.5 for its sampled latent.
    A value of 1.0 is the unmodified explicit-Euler baseline.
    """
    scalar = float(scalar)
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"FLUX latent nudging scalar must be finite and positive, got {scalar}.")
    return latent * scalar


@torch.no_grad()
def encode_flux_prompt(
    pipe: FluxPipeline,
    *,
    prompt: str,
    max_sequence_length: int,
    device: torch.device | str,
) -> FluxConditioning:
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }


@torch.no_grad()
def prepare_flux_latents(
    pipe: FluxPipeline,
    *,
    height: int,
    width: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_channels_latents = int(pipe.transformer.config.in_channels) // 4
    return pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=pipe.transformer.dtype,
        device=device,
        generator=generator,
    )


@torch.no_grad()
def predict_flux_velocity(
    pipe: FluxPipeline,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: FluxConditioning,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    timestep_batch = timestep.expand(latents.shape[0]).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    guidance = None
    if bool(pipe.transformer.config.guidance_embeds):
        guidance = torch.full(
            (latents.shape[0],),
            float(guidance_scale),
            device=latents.device,
            dtype=torch.float32,
        )

    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep_batch / 1000,
        guidance=guidance,
        pooled_projections=conditioning["pooled_prompt_embeds"],
        encoder_hidden_states=conditioning["prompt_embeds"],
        txt_ids=conditioning["text_ids"],
        img_ids=latent_image_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]


@torch.no_grad()
def sample_flux_latent(
    pipe: FluxPipeline,
    *,
    initial_latents: torch.Tensor,
    conditioning: FluxConditioning,
    latent_image_ids: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    save_trajectory: bool,
    save_velocities: bool,
    progress_desc: str | None = None,
) -> tuple[
    torch.Tensor,
    FluxSchedule,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Sample a packed FLUX latent and optionally retain the complete path."""
    schedule = prepare_flux_schedule(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        image_seq_len=int(initial_latents.shape[1]),
        device=initial_latents.device,
    )
    latents = initial_latents
    trajectory = [latents.detach().cpu()] if save_trajectory else []
    velocities: list[torch.Tensor] = []

    for step_idx, timestep in enumerate(
        tqdm(
            schedule.timesteps,
            desc=progress_desc,
            leave=False,
            disable=progress_desc is None,
        )
    ):
        velocity = predict_flux_velocity(
            pipe,
            latents=latents,
            timestep=timestep,
            conditioning=conditioning,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
        )
        latents = pipe.scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=latents,
            return_dict=False,
        )[0]
        if save_trajectory:
            trajectory.append(latents.detach().cpu())
        if save_velocities:
            velocities.append(velocity.detach().cpu())

    return latents.detach(), schedule, trajectory, velocities


@torch.no_grad()
def invert_flux_latent(
    pipe: FluxPipeline,
    *,
    final_latent: torch.Tensor,
    schedule: FluxSchedule,
    conditioning: FluxConditioning,
    latent_image_ids: torch.Tensor,
    guidance_scale: float,
    solver: str = "euler",
    fixed_point_refinement_steps: int = 1,
    save_trajectory: bool,
    save_velocities: bool,
    progress_desc: str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Invert a packed latent with Euler, fixed-point refinement, or Heun.

    ``fixed_point`` repeatedly evaluates the noisy-endpoint velocity and substitutes it
    into the inverse of the corresponding sampling Euler step. A value of one refinement
    therefore costs two NFE per interval. It is an intentionally minimal fixed-point
    control, not the complete ReNoise algorithm.

    ``heun`` evaluates the vector field at both endpoints of the reverse interval and
    applies their trapezoidal average. It also costs two NFE per interval.

    When velocities are retained, the list stores one effective update velocity per
    interval: the final fixed-point velocity or the averaged Heun velocity.
    """
    solver = validate_flux_inversion_solver(solver, fixed_point_refinement_steps)
    latents = final_latent
    trajectory = [latents.detach().cpu()] if save_trajectory else []
    velocities: list[torch.Tensor] = []
    step_indices = range(schedule.num_steps - 1, -1, -1)

    for step_idx in tqdm(
        step_indices,
        total=schedule.num_steps,
        desc=progress_desc,
        leave=False,
        disable=progress_desc is None,
    ):
        clean_latents = latents
        noisy_timestep = schedule.timesteps[step_idx]
        sigma_noisy = schedule.sigmas[step_idx]
        sigma_clean = schedule.sigmas[step_idx + 1]

        if solver == "euler":
            effective_velocity = predict_flux_velocity(
                pipe,
                latents=clean_latents,
                timestep=noisy_timestep,
                conditioning=conditioning,
                latent_image_ids=latent_image_ids,
                guidance_scale=guidance_scale,
            )
            latents = flow_euler_inverse_step(
                clean_latents,
                effective_velocity,
                sigma_noisy=sigma_noisy,
                sigma_clean=sigma_clean,
            )
        elif solver == "fixed_point":
            noisy_estimate = clean_latents
            effective_velocity = None
            for _ in range(1 + int(fixed_point_refinement_steps)):
                effective_velocity = predict_flux_velocity(
                    pipe,
                    latents=noisy_estimate,
                    timestep=noisy_timestep,
                    conditioning=conditioning,
                    latent_image_ids=latent_image_ids,
                    guidance_scale=guidance_scale,
                )
                noisy_estimate = flow_euler_inverse_step(
                    clean_latents,
                    effective_velocity,
                    sigma_noisy=sigma_noisy,
                    sigma_clean=sigma_clean,
                )
            latents = noisy_estimate
        else:
            clean_timestep = (
                schedule.timesteps[step_idx + 1]
                if step_idx + 1 < schedule.num_steps
                else torch.zeros_like(noisy_timestep)
            )
            clean_velocity = predict_flux_velocity(
                pipe,
                latents=clean_latents,
                timestep=clean_timestep,
                conditioning=conditioning,
                latent_image_ids=latent_image_ids,
                guidance_scale=guidance_scale,
            )
            noisy_predictor = flow_euler_inverse_step(
                clean_latents,
                clean_velocity,
                sigma_noisy=sigma_noisy,
                sigma_clean=sigma_clean,
            )
            noisy_velocity = predict_flux_velocity(
                pipe,
                latents=noisy_predictor,
                timestep=noisy_timestep,
                conditioning=conditioning,
                latent_image_ids=latent_image_ids,
                guidance_scale=guidance_scale,
            )
            effective_velocity = (
                0.5 * (clean_velocity.float() + noisy_velocity.float())
            ).to(dtype=clean_velocity.dtype)
            latents = flow_euler_inverse_step(
                clean_latents,
                effective_velocity,
                sigma_noisy=sigma_noisy,
                sigma_clean=sigma_clean,
            )
        if save_trajectory:
            trajectory.append(latents.detach().cpu())
        if save_velocities:
            velocities.append(effective_velocity.detach().cpu())

    return latents.detach(), trajectory, velocities


def oracle_invert_flux_latent(
    *,
    final_latent: torch.Tensor,
    schedule: FluxSchedule,
    sampling_velocities: list[torch.Tensor],
) -> torch.Tensor:
    """Reverse a sampled trajectory using its cached teacher velocities."""
    if len(sampling_velocities) != schedule.num_steps:
        raise ValueError(
            "Oracle inversion needs one sampling velocity per schedule step: "
            f"got {len(sampling_velocities)} for {schedule.num_steps} steps."
        )

    latents = final_latent
    for step_idx in range(schedule.num_steps - 1, -1, -1):
        velocity = sampling_velocities[step_idx].to(
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = flow_euler_inverse_step(
            latents,
            velocity,
            sigma_noisy=schedule.sigmas[step_idx],
            sigma_clean=schedule.sigmas[step_idx + 1],
        )
    return latents.detach()


def unpack_flux_latents(
    pipe: FluxPipeline,
    packed_latents: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    return pipe._unpack_latents(
        packed_latents,
        height,
        width,
        pipe.vae_scale_factor,
    )


@torch.no_grad()
def decode_flux_latents(
    pipe: FluxPipeline,
    packed_latents: torch.Tensor,
    *,
    height: int,
    width: int,
) -> Image.Image:
    latents = unpack_flux_latents(
        pipe,
        packed_latents,
        height=height,
        width=width,
    )
    scaling_factor = float(pipe.vae.config.scaling_factor)
    shift_factor = float(getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0)
    latents = (latents / scaling_factor) + shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
