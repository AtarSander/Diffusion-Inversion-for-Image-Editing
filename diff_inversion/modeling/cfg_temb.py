"""Continuous CFG conditioning injected into the SDXL timestep embedding.

The conditioner keeps the pretrained UNet architecture unchanged. A small MLP
maps the per-sample guidance scale to the timestep-embedding dimension, and a
forward hook adds that vector to ``unet.time_embedding``. The final projection
is zero-initialized, so enabling the conditioner initially preserves the plain
LoRA prediction exactly.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch
from torch import nn

CFG_TEMB_CONDITIONED_KEY = "cfg_temb_conditioned"
CFG_TEMB_CONFIG_KEY = "cfg_temb_config"
CFG_TEMB_STATE_KEY = "cfg_temb_state_dict"
CONDITIONING_TYPE_KEY = "conditioning_type"
CHECKPOINT_FORMAT_VERSION_KEY = "checkpoint_format_version"
LORA_STATE_KEY = "lora_state_dict"

CFG_TEMB_CONDITIONING_TYPE = "cfg_temb"
CFG_TEMB_CHECKPOINT_FORMAT_VERSION = 2
CFG_TEMB_CONDITIONING_MODES = frozenset(
    {
        "cfg_temb",
        "cfg_temb_pair",
        "cfg_temb_branch",
        "cfg_temb_branch_pair",
    }
)

_CONDITIONER_MODULE_NAME = "_cfg_temb_conditioner"


class CfgTembConditioner(nn.Module):
    """Map a continuous CFG value to an additive SDXL timestep embedding."""

    def __init__(
        self,
        *,
        temb_dim: int,
        hidden_dim: int = 128,
        log_mean: float = 1.51749664,
        log_std: float = 0.53920626,
    ):
        super().__init__()
        if int(temb_dim) <= 0:
            raise ValueError(f"cfg temb_dim must be positive, got {temb_dim}.")
        if int(hidden_dim) <= 0:
            raise ValueError(f"cfg hidden_dim must be positive, got {hidden_dim}.")
        if not torch.isfinite(torch.tensor(float(log_mean))):
            raise ValueError(f"cfg log_mean must be finite, got {log_mean}.")
        if not torch.isfinite(torch.tensor(float(log_std))) or float(log_std) <= 0.0:
            raise ValueError(f"cfg log_std must be finite and positive, got {log_std}.")

        self.temb_dim = int(temb_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_mean = float(log_mean)
        self.log_std = float(log_std)
        self.mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.temb_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self._guidance_scale: torch.Tensor | None = None
        self._hook: Any | None = None
        self._enabled = True

    @property
    def num_hooks(self) -> int:
        return int(self._hook is not None)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "input_transform": "log_standardized",
            "log_mean": self.log_mean,
            "log_std": self.log_std,
            "hidden_dim": self.hidden_dim,
            "temb_dim": self.temb_dim,
            "zero_init_output": True,
        }

    def validate_checkpoint_config(self, cfg: dict[str, Any]) -> None:
        required_keys = {
            "input_transform",
            "log_mean",
            "log_std",
            "hidden_dim",
            "temb_dim",
            "zero_init_output",
        }
        missing_keys = required_keys - cfg.keys()
        if missing_keys:
            raise ValueError(
                "CFG temb checkpoint config is missing: " + ", ".join(sorted(missing_keys))
            )

        transform = str(cfg["input_transform"])
        if transform != "log_standardized":
            raise ValueError(f"Unsupported CFG temb input_transform in checkpoint: {transform!r}.")
        if cfg["zero_init_output"] is not True:
            raise ValueError("CFG temb checkpoint must use zero_init_output=true.")
        for key, expected in (
            ("hidden_dim", self.hidden_dim),
            ("temb_dim", self.temb_dim),
        ):
            value = int(cfg[key])
            if value != expected:
                raise ValueError(
                    f"CFG temb checkpoint {key}={value} does not match configured "
                    f"{key}={expected}."
                )
        for key, expected in (
            ("log_mean", self.log_mean),
            ("log_std", self.log_std),
        ):
            value = float(cfg[key])
            if not torch.isclose(
                torch.tensor(value),
                torch.tensor(expected),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(
                    f"CFG temb checkpoint {key}={value} does not match configured "
                    f"{key}={expected}."
                )

    def install_hook(self, time_embedding: nn.Module) -> None:
        self.remove_hook()
        self._hook = time_embedding.register_forward_hook(self._add_embedding_hook)

    def remove_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @contextmanager
    def condition(self, guidance_scale: torch.Tensor | float | int) -> Iterator[None]:
        previous = self._guidance_scale
        self._guidance_scale = self._as_guidance_tensor(guidance_scale)
        try:
            yield
        finally:
            self._guidance_scale = previous

    def embedding_for(
        self,
        guidance_scale: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        guidance_scale = _expand_guidance_batch(guidance_scale, batch_size)
        parameter = self.mlp[0].weight
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                "CFG temb conditioner must stay in float32. Install it after converting "
                "the UNet to its inference dtype."
            )
        values = guidance_scale.to(device=parameter.device, dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("CFG temb guidance values must be finite.")
        if torch.any(values <= 0.0):
            minimum = float(values.detach().min().cpu())
            raise ValueError(f"CFG temb guidance values must be positive, got {minimum}.")

        feature = ((torch.log(values) - self.log_mean) / self.log_std).unsqueeze(-1)
        embedding = self.mlp(feature)
        return embedding.to(device=device, dtype=dtype)

    def _add_embedding_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        if not self._enabled or self._guidance_scale is None or not torch.is_tensor(output):
            return output
        if output.ndim != 2 or int(output.shape[-1]) != self.temb_dim:
            raise ValueError(
                "Unexpected SDXL timestep embedding shape for CFG conditioning: "
                f"got {tuple(output.shape)}, expected [batch, {self.temb_dim}]."
            )
        cfg_embedding = self.embedding_for(
            self._guidance_scale,
            batch_size=int(output.shape[0]),
            device=output.device,
            dtype=output.dtype,
        )
        return output + cfg_embedding

    @staticmethod
    def _as_guidance_tensor(guidance_scale: torch.Tensor | float | int) -> torch.Tensor:
        if torch.is_tensor(guidance_scale):
            return guidance_scale.detach().flatten()
        return torch.tensor([float(guidance_scale)], dtype=torch.float32)


def _expand_guidance_batch(guidance_scale: torch.Tensor, batch_size: int) -> torch.Tensor:
    values = guidance_scale.flatten()
    if values.numel() == 1:
        return values.expand(batch_size)
    if values.numel() == batch_size:
        return values
    if batch_size == 2 * values.numel():
        return torch.cat([values, values], dim=0)
    raise ValueError(
        "CFG temb batch size does not match timestep embedding output: "
        f"got {values.numel()} CFG values for batch_size={batch_size}."
    )


def _time_embedding_output_dim(unet: nn.Module) -> int:
    time_embedding = getattr(unet, "time_embedding", None)
    if time_embedding is None:
        raise AttributeError("UNet does not expose time_embedding required for CFG conditioning.")
    linear_2 = getattr(time_embedding, "linear_2", None)
    output_dim = getattr(linear_2, "out_features", None)
    if output_dim is None:
        raise AttributeError(
            "UNet time_embedding does not expose linear_2.out_features; "
            "cannot infer CFG temb dimension."
        )
    return int(output_dim)


def install_cfg_temb_conditioner(
    unet: nn.Module,
    *,
    hidden_dim: int,
    log_mean: float,
    log_std: float,
) -> CfgTembConditioner:
    if get_cfg_temb_conditioner(unet) is not None:
        raise RuntimeError("CFG timestep-embedding conditioner is already installed.")

    temb_dim = _time_embedding_output_dim(unet)
    conditioner = CfgTembConditioner(
        temb_dim=temb_dim,
        hidden_dim=hidden_dim,
        log_mean=log_mean,
        log_std=log_std,
    )
    reference_parameter = next(unet.time_embedding.parameters())
    conditioner.to(device=reference_parameter.device, dtype=torch.float32)
    unet.add_module(_CONDITIONER_MODULE_NAME, conditioner)
    conditioner.install_hook(unet.time_embedding)
    return conditioner


def get_cfg_temb_conditioner(unet: nn.Module) -> CfgTembConditioner | None:
    value = getattr(unet, _CONDITIONER_MODULE_NAME, None)
    return value if isinstance(value, CfgTembConditioner) else None


def cfg_temb_context(
    unet: nn.Module,
    guidance_scale: torch.Tensor | float | int | None,
):
    conditioner = get_cfg_temb_conditioner(unet)
    if conditioner is None or guidance_scale is None:
        return nullcontext()
    return conditioner.condition(guidance_scale)


def set_cfg_temb_enabled(unet: nn.Module, enabled: bool) -> bool:
    conditioner = get_cfg_temb_conditioner(unet)
    if conditioner is None:
        return False
    conditioner.set_enabled(enabled)
    return True


def split_cfg_temb_checkpoint_state(
    state: Any,
) -> tuple[Any, dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    """Return nested LoRA state plus optional CFG-temb state and config."""
    cfg_state, cfg_config = extract_cfg_temb_checkpoint_metadata(state)
    if cfg_state is None:
        return state, None, None
    if LORA_STATE_KEY not in state:
        raise ValueError("CFG temb checkpoint is missing lora_state_dict.")
    return state[LORA_STATE_KEY], cfg_state, cfg_config


def extract_cfg_temb_checkpoint_metadata(
    state: Any,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    """Validate and return CFG-temb parameters stored alongside any checkpoint payload."""
    if not isinstance(state, dict):
        return None, None
    metadata_keys = {
        CFG_TEMB_CONDITIONED_KEY,
        CONDITIONING_TYPE_KEY,
        CHECKPOINT_FORMAT_VERSION_KEY,
        CFG_TEMB_STATE_KEY,
        CFG_TEMB_CONFIG_KEY,
    }
    if metadata_keys.isdisjoint(state):
        return None, None
    if state.get(CFG_TEMB_CONDITIONED_KEY) is not True:
        raise ValueError("CFG temb checkpoint must set cfg_temb_conditioned=true.")
    if state.get(CONDITIONING_TYPE_KEY) != CFG_TEMB_CONDITIONING_TYPE:
        raise ValueError(
            f"Unsupported checkpoint conditioning_type={state.get(CONDITIONING_TYPE_KEY)!r}; "
            f"expected {CFG_TEMB_CONDITIONING_TYPE!r}."
        )
    checkpoint_version = state.get(CHECKPOINT_FORMAT_VERSION_KEY)
    if checkpoint_version != CFG_TEMB_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported CFG temb checkpoint format version {checkpoint_version!r}; "
            f"expected {CFG_TEMB_CHECKPOINT_FORMAT_VERSION}."
        )
    cfg_state = state.get(CFG_TEMB_STATE_KEY)
    if not isinstance(cfg_state, dict):
        raise ValueError("CFG temb checkpoint is missing cfg_temb_state_dict.")
    cfg_config = state.get(CFG_TEMB_CONFIG_KEY)
    if not isinstance(cfg_config, dict):
        raise ValueError("CFG temb checkpoint is missing cfg_temb_config.")
    return cfg_state, cfg_config


def make_cfg_temb_checkpoint_state(
    *,
    lora_state_dict: dict[str, Any],
    conditioner: CfgTembConditioner | None,
) -> dict[str, Any]:
    if conditioner is None:
        return lora_state_dict
    return {
        **cfg_temb_checkpoint_metadata(conditioner),
        LORA_STATE_KEY: lora_state_dict,
    }


def cfg_temb_checkpoint_metadata(
    conditioner: CfgTembConditioner | None,
) -> dict[str, Any]:
    if conditioner is None:
        return {}
    return {
        CHECKPOINT_FORMAT_VERSION_KEY: CFG_TEMB_CHECKPOINT_FORMAT_VERSION,
        CONDITIONING_TYPE_KEY: CFG_TEMB_CONDITIONING_TYPE,
        CFG_TEMB_CONDITIONED_KEY: True,
        CFG_TEMB_STATE_KEY: conditioner.state_dict(),
        CFG_TEMB_CONFIG_KEY: conditioner.checkpoint_config(),
    }
