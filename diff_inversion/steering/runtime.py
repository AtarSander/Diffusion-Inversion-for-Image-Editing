"""DDIM/MW inversion, base-model activation reads, and fresh-noise generation."""

import copy
from contextlib import contextmanager

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

from diff_inversion.modeling.sdxl_sampling import predict_noise_sdxl

from .hooks import feature_hooks


@contextmanager
def pipeline_state(pipe, *, lora=False):
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
    def __init__(self, pipe, *, steps=50):
        self.pipe, self.steps = pipe, steps
        self.scheduler_config = dict(pipe.scheduler.config)
        sampling = self.scheduler()
        inverse = self.scheduler(inverse=True)
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

    def scheduler(self, *, inverse=False):
        cls = DDIMInverseScheduler if inverse else DDIMScheduler
        scheduler = cls.from_config(copy.deepcopy(self.scheduler_config))
        scheduler.set_timesteps(self.steps, device=self.pipe.device)
        return scheduler

    @torch.no_grad()
    def invert(self, latent, conditioning, method):
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
    def read(self, states, conditioning, *, reader="free"):
        """Free base rollout from recovered zT, or independent saved-state probes."""
        if reader not in ("free", "direct") or len(states) != self.steps + 1:
            raise ValueError("Expected all inverse states and reader=free/direct")
        features, moments = {}, {}
        with pipeline_state(self.pipe):
            scheduler = self.scheduler()
            self.pipe.scheduler = scheduler
            current = states[-1].to(device=self.pipe.device, dtype=self.pipe.unet.dtype)
            for call, timestep in enumerate(scheduler.timesteps):
                if reader == "direct":
                    current = states[self.steps - call].to(
                        device=self.pipe.device, dtype=self.pipe.unet.dtype
                    )
                capture, second = {}, {}
                with feature_hooks(self.pipe.unet, captures=capture, moments=second):
                    noise = predict_noise_sdxl(self.pipe, current, timestep, conditioning, 1.0)
                for rep in capture:
                    for name, value in capture[rep].items():
                        features.setdefault(rep, {}).setdefault(name, []).append(value[0])
                        moments.setdefault(rep, {}).setdefault(name, []).append(
                            second[rep][name][0]
                        )
                if reader == "free":
                    current = scheduler.step(noise, timestep, current, eta=0).prev_sample
        return {
            "features": {
                rep: {name: torch.stack(value) for name, value in layers.items()}
                for rep, layers in features.items()
            },
            "moments": {
                rep: {name: torch.stack(value) for name, value in layers.items()}
                for rep, layers in moments.items()
            },
            "reconstruction": current.detach().cpu() if reader == "free" else None,
        }

    @torch.no_grad()
    def generate(
        self,
        starting_noise,
        conditioning,
        *,
        guidance_scale=1.0,
        representation=None,
        offsets=None,
    ):
        with pipeline_state(self.pipe):
            scheduler = self.scheduler()
            self.pipe.scheduler = scheduler
            current = starting_noise.clone().to(
                device=self.pipe.device, dtype=self.pipe.unet.dtype
            )
            for call, timestep in enumerate(scheduler.timesteps):
                with feature_hooks(
                    self.pipe.unet, representation=representation, deltas=(offsets or {}).get(call)
                ):
                    noise = predict_noise_sdxl(
                        self.pipe, current, timestep, conditioning, guidance_scale
                    )
                current = scheduler.step(noise, timestep, current, eta=0).prev_sample
        return current.detach().cpu()
