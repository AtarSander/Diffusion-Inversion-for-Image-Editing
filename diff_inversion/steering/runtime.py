"""DDIM/MW inversion, activation reads at inverse states, and fresh-noise generation."""

import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl

from .hooks import apply_offsets, capture_features
from .types import ActivationRead, Conditioning, Method, Offsets, Representation


@contextmanager
def pipeline_state(pipe: StableDiffusionXLPipeline, *, lora: bool = False) -> Iterator[None]:
    """Keep adapters on only for MW inversion; restore scheduler/flags on failure."""
    scheduler = pipe.scheduler
    adapters = [
        module
        for module in pipe.unet.modules()
        if module is not pipe.unet and hasattr(module, "enable_adapters")
    ]
    if lora and not adapters:
        raise ValueError("MW inversion requires a loaded LoRA checkpoint")
    flags = [module.disable_adapters for module in adapters]
    # PEFT's enable_adapters also changes requires_grad, so restore both kinds of flags.
    trainability = [(parameter, parameter.requires_grad) for parameter in pipe.unet.parameters()]
    try:
        for module in adapters:
            module.enable_adapters(lora)
        yield
    finally:
        pipe.scheduler = scheduler
        for module, disabled in zip(adapters, flags):
            module.enable_adapters(not disabled)
        for parameter, requires_grad in trainability:
            parameter.requires_grad_(requires_grad)


class SteeringRuntime:
    def __init__(self, pipe: StableDiffusionXLPipeline, *, steps: int = 50) -> None:
        self.pipe = pipe
        self.steps = steps
        self.scheduler_config = dict(pipe.scheduler.config)
        sampling = self.scheduler()
        inverse = self.scheduler(inverse=True)
        # Entry 0 is the encoded image; each later entry is an inverse-step output.
        self.state_timesteps: list[int | None] = [None, *inverse.timesteps.tolist()]
        # This reader indexes inverse states N-i. It requires matching arithmetic
        # predecessors, not just similarly named scheduler timesteps.
        ratio = sampling.config.num_train_timesteps // steps
        grid = sampling.timesteps.tolist()
        if (
            sampling.config.prediction_type != "epsilon"
            or sampling.config.clip_sample
            or grid[::-1] != inverse.timesteps.tolist()
            or any(a - b != ratio for a, b in zip(grid, grid[1:]))
        ):
            raise ValueError(
                "Use matching, uniform epsilon-prediction DDIM schedules without clipping"
            )

    def scheduler(self, *, inverse: bool = False) -> DDIMScheduler | DDIMInverseScheduler:
        cls = DDIMInverseScheduler if inverse else DDIMScheduler
        scheduler = cls.from_config(copy.deepcopy(self.scheduler_config))
        scheduler.set_timesteps(self.steps, device=self.pipe.device)
        return scheduler

    @torch.no_grad()
    def invert(
        self, latent: torch.Tensor, conditioning: Conditioning, method: Method
    ) -> list[torch.Tensor]:
        if method not in ("DDIM", "MW"):
            raise ValueError(f"Unknown inversion method: {method}")
        with pipeline_state(self.pipe, lora=method == "MW"):
            scheduler = self.scheduler(inverse=True)
            self.pipe.scheduler = scheduler
            current = latent.clone()
            states = [current.detach().cpu()]
            for timestep in scheduler.timesteps:
                noise = predict_noise_sdxl(self.pipe, current, timestep, conditioning, 1.0)
                current = scheduler.step(noise, timestep, current).prev_sample
                states.append(current.detach().cpu())
        return states

    @torch.no_grad()
    def read_inversion_states(
        self, states: Sequence[torch.Tensor], conditioning: Conditioning
    ) -> ActivationRead:
        """One base-UNet forward per saved inverse state, in denoising order.

        For N steps, call k uses states[N-k] at its matching diffusion timestep.
        No predicted state is propagated to another read; states[0] is the clean
        endpoint and is not a denoising input. LoRA is off for every feature read.
        """
        if len(states) != self.steps + 1:
            raise ValueError("Expected the encoded image and every inverse-step output")
        features: dict[Representation, dict[str, list[torch.Tensor]]] = {}
        with pipeline_state(self.pipe):
            scheduler = self.scheduler()
            self.pipe.scheduler = scheduler
            for state, timestep in zip(reversed(states[1:]), scheduler.timesteps):
                current = state.to(device=self.pipe.device, dtype=self.pipe.unet.dtype)
                with capture_features(self.pipe.unet) as capture:
                    predict_noise_sdxl(self.pipe, current, timestep, conditioning, 1.0)
                for rep in capture:
                    for name, value in capture[rep].items():
                        features.setdefault(rep, {}).setdefault(name, []).append(value[0])
        return {
            "features": {
                rep: {name: torch.stack(value) for name, value in layers.items()}
                for rep, layers in features.items()
            },
        }

    @torch.no_grad()
    def generate(
        self,
        starting_noise: torch.Tensor,
        conditioning: Conditioning,
        *,
        guidance_scale: float = 1.0,
        representation: Representation | None = None,
        offsets: Offsets | None = None,
        renorm: bool = False,
    ) -> torch.Tensor:
        with pipeline_state(self.pipe):
            scheduler = self.scheduler()
            self.pipe.scheduler = scheduler
            current = starting_noise.clone().to(
                device=self.pipe.device, dtype=self.pipe.unet.dtype
            )
            for call, timestep in enumerate(scheduler.timesteps):
                with apply_offsets(
                    self.pipe.unet,
                    representation,
                    (offsets or {}).get(call, {}),
                    renorm=renorm,
                ):
                    noise = predict_noise_sdxl(
                        self.pipe, current, timestep, conditioning, guidance_scale
                    )
                current = scheduler.step(noise, timestep, current, eta=0).prev_sample
        return current.detach().cpu()
