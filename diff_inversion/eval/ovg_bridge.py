"""Pinned OVG imports and our trajectory-supervised inversion method."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

OVG_REVISION = "e13fe392ba8f0738ce2f041ee356bb2b5a0171db"


def load_upstream(root: Path) -> None:
    root = root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OVG_REVISION:
        raise ValueError(f"Expected OVG {OVG_REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("The pinned OVG checkout contains local modifications")
    if "ovg" in sys.modules:
        if Path(sys.modules["ovg"].__file__).resolve() != root / "__init__.py":
            raise RuntimeError("Another ovg package is already imported")
        return
    # Upstream sources live at repository root, although they import ovg.*.
    spec = importlib.util.spec_from_file_location(
        "ovg", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ovg"] = module
    spec.loader.exec_module(module)


def make_inversion_method(use_lora: bool):
    import torch
    from diffusers import DDIMInverseScheduler
    from ovg.inversion_methods import StandardInversion
    from ovg.utils.predict import predict_denoiser

    class TrajectoryInversion(StandardInversion):
        def __init__(self):
            super().__init__()
            self.name = "lora_inversion" if use_lora else "matched_ddim"

        @torch.no_grad()
        def invert(self, image, denoiser, scheduler, encoder_hidden_states,
                   num_steps=50, **kwargs):
            if scheduler.config.prediction_type != "epsilon":
                raise ValueError("Our current inversion adapters require epsilon prediction")
            inverse = DDIMInverseScheduler.from_config(scheduler.config)
            inverse.set_timesteps(num_steps, device=image.device)
            latents = self._match_denoiser(image, denoiser, copy=True)
            condition = self._ensure_condition_dtype(encoder_hidden_states, denoiser)
            states, noises, times = [latents.cpu()], [], []
            capture = kwargs.get("store_intermediates", False)
            if use_lora:
                denoiser.enable_adapters()
            try:
                for timestep in inverse.timesteps:
                    eps = predict_denoiser(denoiser, latents, timestep, cond=condition)
                    latents = inverse.step(eps, timestep, latents).prev_sample
                    if capture:
                        states.append(latents.cpu())
                    if kwargs.get("store_predicted_noise", False):
                        noises.append(eps.cpu())
                    times.append(int(timestep))
            finally:
                if use_lora:
                    denoiser.disable_adapters()
            return {"latents": latents, "intermediate_latents": states if capture else [],
                    "predicted_noises": noises, "timesteps": times}

    return TrajectoryInversion()
