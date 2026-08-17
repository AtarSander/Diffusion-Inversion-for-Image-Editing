"""Run FLUX sampling, explicit-Euler inversion, and reconstruction in one command.

This evaluator is project code. Its 50-step FLUX inversion and normality/reconstruction
comparison follows Appendix P of *There and Back Again* (arXiv:2410.23530v5). The
optional latent-scaling controls are motivated by Stable Flow and the UniEdit-Flow
script bundled with the TABA repository. This is not a Stable Flow implementation:
Stable Flow's feature injection is not used. Shared TABA-derived metric modules carry
their own source notices. See ``docs/flow_matching_provenance.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.data.generate_flux_inversion_data import (
    build_flux_inversion_pairs,
    load_prompt_records,
    save_flux_inversion_training_sample,
)
from diff_inversion.eval.clip_alignment import add_clip_text_alignment_metrics
from diff_inversion.eval.correlation import get_top_k_corr_in_patches
from diff_inversion.eval.normality import kl_div, kl_div2
from diff_inversion.eval.reporting import flatten_metrics, summarize_numeric_rows
from diff_inversion.eval.sample_metrics import (
    image_pair_metrics,
    load_rgb_tensor,
    noise_normality_metrics,
    pair_metrics,
    plain_area_mask,
    tensor_stats,
)
from diff_inversion.modeling.flux_sampling import (
    FluxSchedule,
    decode_flux_latents,
    encode_flux_prompt,
    flux_inversion_nfe,
    invert_flux_latent,
    nudge_flux_latent,
    oracle_invert_flux_latent,
    prepare_flux_latents,
    sample_flux_latent,
    unpack_flux_latents,
    validate_flux_inversion_solver,
)
from diff_inversion.utils import resolve_torch_dtype

TRAINING_CACHE_FILES = (
    "conditioning.pt",
    "latent_image_ids.pt",
    "timesteps.pt",
    "inversion_inputs.pt",
    "target_velocities.pt",
    "meta.json",
)

SOURCE_SAMPLE_FILES = (
    "metrics.json",
    "prompt.json",
    "original.png",
    "initial_noise.pt",
    "final_latent.pt",
    "oracle_inverted_noise.pt",
    "timesteps.pt",
    "sigmas.pt",
)


@dataclass(frozen=True)
class FluxGuidanceScales:
    """Embedded-guidance values used at each stage of a FLUX round trip."""

    source: float
    inversion: float
    reconstruction: float
    editing: float


def resolve_flux_guidance_scales(cfg: DictConfig) -> FluxGuidanceScales:
    """Resolve optional stage overrides while preserving the legacy matched protocol."""
    source = float(cfg.guidance_scale)
    inversion_override = OmegaConf.select(cfg, "inversion.guidance_scale", default=None)
    reconstruction_override = OmegaConf.select(
        cfg,
        "reconstruction.guidance_scale",
        default=None,
    )
    editing_override = OmegaConf.select(cfg, "editing.guidance_scale", default=None)
    inversion = source if inversion_override is None else float(inversion_override)
    reconstruction = (
        source if reconstruction_override is None else float(reconstruction_override)
    )
    editing = reconstruction if editing_override is None else float(editing_override)
    return FluxGuidanceScales(
        source=source,
        inversion=inversion,
        reconstruction=reconstruction,
        editing=editing,
    )


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _json_default(value: Any):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON.")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)


def build_flux_prompt_jobs(
    prompts: list[str],
    *,
    samples_per_prompt: int,
    start_index: int,
    num_samples: int | None,
) -> list[tuple[int, str]]:
    if not prompts:
        raise ValueError("At least one FLUX prompt is required.")
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive.")
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    expanded = [prompt for prompt in prompts for _ in range(samples_per_prompt)]
    available = len(expanded) - start_index
    selected_count = available if num_samples is None else int(num_samples)
    if selected_count <= 0 or selected_count > available:
        raise ValueError(
            f"Requested {selected_count} samples from start_index={start_index}, "
            f"but only {max(available, 0)} prompt jobs are available."
        )
    stop_index = start_index + selected_count
    return [(index, expanded[index]) for index in range(start_index, stop_index)]


def build_flux_edit_jobs(
    prompt_jobs: list[tuple[int, str]],
    *,
    target_prompts: list[str],
    target_prompt_template: str | None,
    target_prompt_offset: int,
) -> list[tuple[int, str, str | None]]:
    """Attach deterministic target prompts to source-prompt jobs."""
    if target_prompt_template and target_prompts:
        raise ValueError(
            "Configure either editing.target_prompt_template or "
            "editing.target_prompts_jsonl, not both."
        )
    if target_prompt_template:
        if "{prompt}" not in target_prompt_template:
            raise ValueError(
                "editing.target_prompt_template must contain the {prompt} placeholder."
            )
        return [
            (sample_index, prompt, target_prompt_template.format(prompt=prompt))
            for sample_index, prompt in prompt_jobs
        ]
    if target_prompts:
        offset = int(target_prompt_offset)
        return [
            (
                sample_index,
                prompt,
                target_prompts[(sample_index + offset) % len(target_prompts)],
            )
            for sample_index, prompt in prompt_jobs
        ]
    return [(sample_index, prompt, None) for sample_index, prompt in prompt_jobs]


def _masked_mean_abs(tensor: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(tensor[:, mask].abs().mean().item())


def _masked_std(tensor: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(tensor[:, mask].std(unbiased=False).item())


def masked_flux_noise_region_metrics(
    image_path: Path,
    initial_noise: torch.Tensor,
    inverted_noise: torch.Tensor,
    *,
    plain_threshold: float,
) -> dict[str, float | None]:
    """Appendix-P-style error and latent std in plain/non-plain image regions."""
    if initial_noise.ndim == 4 and initial_noise.shape[0] == 1:
        initial_noise = initial_noise[0]
    if inverted_noise.ndim == 4 and inverted_noise.shape[0] == 1:
        inverted_noise = inverted_noise[0]
    if initial_noise.ndim != 3 or inverted_noise.shape != initial_noise.shape:
        raise ValueError(
            "Expected matching [C,H,W] unpacked FLUX latents, got "
            f"{tuple(initial_noise.shape)} and {tuple(inverted_noise.shape)}."
        )

    image = load_rgb_tensor(
        image_path,
        size=(int(initial_noise.shape[-1]), int(initial_noise.shape[-2])),
    )
    plain_mask = plain_area_mask(image, threshold=plain_threshold)
    non_plain_mask = ~plain_mask
    delta = inverted_noise.float() - initial_noise.float()
    return {
        "plain_abs_error": _masked_mean_abs(delta, plain_mask),
        "non_plain_abs_error": _masked_mean_abs(delta, non_plain_mask),
        "plain_initial_noise_std": _masked_std(initial_noise.float(), plain_mask),
        "non_plain_initial_noise_std": _masked_std(initial_noise.float(), non_plain_mask),
        "plain_inverted_noise_std": _masked_std(inverted_noise.float(), plain_mask),
        "non_plain_inverted_noise_std": _masked_std(inverted_noise.float(), non_plain_mask),
        "plain_pixel_fraction": float(plain_mask.float().mean().item()),
    }


def _nested_get(value: dict[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _validate_existing_run_config(output_dir: Path, cfg: DictConfig) -> None:
    path = output_dir / "run_config.yaml"
    if not path.exists():
        return
    existing = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    current = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(existing, dict) or not isinstance(current, dict):
        raise TypeError(f"Expected mapping configs while validating {path}.")
    checked_keys = (
        "prompts",
        "prompts_jsonl",
        "prompt_key",
        "samples_per_prompt",
        "seed",
        "dataset_split",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "inversion.guidance_scale",
        "inversion.latent_nudging_scalar",
        "inversion.solver",
        "inversion.fixed_point_refinement_steps",
        "reconstruction.guidance_scale",
        "max_sequence_length",
        "model.model_id",
        "model.torch_dtype",
        "model.revision",
        "model.variant",
        "lora",
        "editing",
        "save.training_data",
        "save.storage_dtype",
        "metrics",
    )
    mismatches = [
        (key, _nested_get(existing, key), _nested_get(current, key))
        for key in checked_keys
        if _nested_get(existing, key) != _nested_get(current, key)
    ]
    if mismatches:
        details = "\n".join(
            f"  - {key}: existing={old!r}, current={new!r}" for key, old, new in mismatches
        )
        raise ValueError(
            f"Refusing to mix incompatible FLUX round-trip runs in {output_dir}:\n"
            f"{details}\nUse a new output_dir."
        )


def _validate_source_run_config(source_dir: Path, cfg: DictConfig) -> None:
    path = source_dir / "run_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"FLUX source run config not found: {path}")
    source = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    current = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(source, dict) or not isinstance(current, dict):
        raise TypeError(f"Expected mapping configs while validating {path}.")
    checked_keys = (
        "prompts",
        "prompts_jsonl",
        "prompt_key",
        "samples_per_prompt",
        "seed",
        "dataset_split",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "model.model_id",
        "model.torch_dtype",
        "model.revision",
        "model.variant",
    )
    mismatches = [
        (key, _nested_get(source, key), _nested_get(current, key))
        for key in checked_keys
        if _nested_get(source, key) != _nested_get(current, key)
    ]
    if mismatches:
        details = "\n".join(
            f"  - {key}: source={old!r}, current={new!r}" for key, old, new in mismatches
        )
        raise ValueError(
            f"FLUX source run {source_dir} is incompatible with this evaluation:\n"
            f"{details}"
        )


def _validate_source_sample(
    source_sample_dir: Path,
    *,
    expected_prompt: str,
    expected_seed: int,
) -> None:
    missing = [name for name in SOURCE_SAMPLE_FILES if not (source_sample_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete FLUX source sample {source_sample_dir}; missing: {missing}"
        )
    with (source_sample_dir / "prompt.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if str(metadata.get("prompt")) != expected_prompt:
        raise ValueError(
            f"Prompt mismatch in {source_sample_dir}: "
            f"expected {expected_prompt!r}, found {metadata.get('prompt')!r}."
        )
    if int(metadata.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Seed mismatch in {source_sample_dir}: "
            f"expected {expected_seed}, found {metadata.get('seed')!r}."
        )


def _validate_source_noise_layout(
    source_sample_dir: Path,
    stored_noise: torch.Tensor,
    prepared_noise: torch.Tensor,
) -> None:
    """Validate layout without assuming cross-device CUDA RNG bit equality.

    Prompt and seed identity are checked from the persisted source metadata. The stored
    tensor is the scientific control and must be reused exactly; regenerating it only to
    compare values would incorrectly assume identical CUDA RNG streams across devices.
    """
    if stored_noise.shape != prepared_noise.shape:
        raise ValueError(
            f"Initial-noise shape mismatch in {source_sample_dir}: "
            f"stored={tuple(stored_noise.shape)}, expected={tuple(prepared_noise.shape)}."
        )


def _link_source_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    if destination.is_symlink():
        if destination.resolve(strict=False) == source.resolve():
            return
        raise FileExistsError(f"Conflicting symlink already exists: {destination}")
    destination.symlink_to(source.resolve())


def _sample_is_complete(
    sample_dir: Path,
    *,
    require_training_data: bool,
    require_editing: bool = False,
    require_initial_noise_edit: bool = False,
) -> bool:
    if not (sample_dir / "metrics.json").exists():
        return False
    if require_training_data:
        if not all((sample_dir / name).exists() for name in TRAINING_CACHE_FILES):
            return False
    if require_editing and not (sample_dir / "edited.png").exists():
        return False
    if require_initial_noise_edit and not (
        sample_dir / "edited_from_initial_noise.png"
    ).exists():
        return False
    return True


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value)!r}.")
    return value


def _resolve_device(cfg: DictConfig) -> torch.device:
    configured = str(cfg.device).strip().lower()
    if configured == "auto":
        configured = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured)
    if bool(cfg.model.require_cuda) and device.type != "cuda":
        raise RuntimeError("FLUX baseline is configured with model.require_cuda=true.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _load_flux_pipeline(cfg: DictConfig, device: torch.device) -> FluxPipeline:
    dtype = resolve_torch_dtype(str(cfg.model.torch_dtype))
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": bool(cfg.model.local_files_only),
    }
    if cfg.model.revision is not None:
        load_kwargs["revision"] = str(cfg.model.revision)
    if cfg.model.variant is not None:
        load_kwargs["variant"] = str(cfg.model.variant)

    logger.info("Loading FLUX pipeline: {}", cfg.model.model_id)
    pipe = FluxPipeline.from_pretrained(str(cfg.model.model_id), **load_kwargs)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if bool(cfg.lora.enabled):
        if cfg.lora.path is None:
            raise ValueError("lora.path is required when lora.enabled=true.")
        lora_path = _resolve_path(str(cfg.lora.path))
        if not lora_path.exists():
            raise FileNotFoundError(f"FLUX inversion LoRA not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path), adapter_name="inversion")
        pipe.set_adapters("inversion", adapter_weights=float(cfg.lora.scale))
        # The adapter is trained for cleaner inversion queries, not normal sampling.
        pipe.disable_lora()
        logger.info("Loaded inversion-only FLUX LoRA from {}", lora_path)

    offload = str(cfg.model.offload).strip().lower()
    if offload == "model":
        if device.type != "cuda":
            raise ValueError("model.offload=model requires a CUDA device.")
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        if device.type != "cuda":
            raise ValueError("model.offload=sequential requires a CUDA device.")
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "none":
        pipe.to(device)
    else:
        raise ValueError(
            f"Unsupported model.offload={cfg.model.offload!r}; expected none, model, or sequential."
        )

    if bool(cfg.model.vae_tiling):
        pipe.vae.enable_tiling()
    logger.success("FLUX pipeline loaded with offload={} on {}", offload, device)
    return pipe


@contextmanager
def _inversion_lora(pipe: FluxPipeline, enabled: bool) -> Iterator[None]:
    if enabled:
        pipe.enable_lora()
    try:
        yield
    finally:
        if enabled:
            pipe.disable_lora()


def _execution_device(pipe: FluxPipeline) -> torch.device:
    return torch.device(pipe._execution_device)


def _stack_if_enabled(tensors: list[torch.Tensor], enabled: bool) -> torch.Tensor | None:
    if not enabled or not tensors:
        return None
    return torch.stack(tensors, dim=0)


def _save_sample_tensors(
    sample_dir: Path,
    *,
    initial_noise: torch.Tensor,
    final_latent: torch.Tensor,
    inversion_start_latent: torch.Tensor,
    inverted_noise: torch.Tensor,
    oracle_inverted_noise: torch.Tensor,
    reconstructed_latent: torch.Tensor,
    schedule,
    sampling_trajectory: list[torch.Tensor],
    sampling_velocities: list[torch.Tensor],
    inversion_trajectory: list[torch.Tensor],
    inversion_velocities: list[torch.Tensor],
    save_trajectories: bool,
    save_velocities: bool,
    save_source_inputs: bool = True,
) -> None:
    if save_source_inputs:
        torch.save(initial_noise.detach().cpu(), sample_dir / "initial_noise.pt")
        torch.save(final_latent.detach().cpu(), sample_dir / "final_latent.pt")
        torch.save(oracle_inverted_noise.detach().cpu(), sample_dir / "oracle_inverted_noise.pt")
        torch.save(schedule.timesteps.detach().cpu(), sample_dir / "timesteps.pt")
        torch.save(schedule.sigmas.detach().cpu(), sample_dir / "sigmas.pt")
    torch.save(
        inversion_start_latent.detach().cpu(),
        sample_dir / "inversion_start_latent.pt",
    )
    torch.save(inverted_noise.detach().cpu(), sample_dir / "inverted_noise.pt")
    torch.save(reconstructed_latent.detach().cpu(), sample_dir / "reconstructed_latent.pt")

    if save_source_inputs:
        sampling_trajectory_tensor = _stack_if_enabled(sampling_trajectory, save_trajectories)
        if sampling_trajectory_tensor is not None:
            torch.save(sampling_trajectory_tensor, sample_dir / "sampling_trajectory.pt")
        sampling_velocity_tensor = _stack_if_enabled(sampling_velocities, save_velocities)
        if sampling_velocity_tensor is not None:
            torch.save(sampling_velocity_tensor, sample_dir / "sampling_velocities.pt")
    inversion_trajectory_tensor = _stack_if_enabled(inversion_trajectory, save_trajectories)
    if inversion_trajectory_tensor is not None:
        torch.save(inversion_trajectory_tensor, sample_dir / "inversion_trajectory.pt")
    inversion_velocity_tensor = _stack_if_enabled(inversion_velocities, save_velocities)
    if inversion_velocity_tensor is not None:
        torch.save(inversion_velocity_tensor, sample_dir / "inversion_velocities.pt")


@torch.no_grad()
def _run_sample(
    pipe: FluxPipeline,
    *,
    sample_dir: Path,
    sample_index: int,
    prompt: str,
    target_prompt: str | None,
    seed: int,
    cfg: DictConfig,
    source_sample_dir: Path | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    device = _execution_device(pipe)
    height = int(cfg.height)
    width = int(cfg.width)
    guidance_scales = resolve_flux_guidance_scales(cfg)
    num_inference_steps = int(cfg.num_inference_steps)
    latent_nudging_scalar = float(cfg.inversion.latent_nudging_scalar)
    inversion_solver = validate_flux_inversion_solver(
        str(cfg.inversion.solver),
        int(cfg.inversion.fixed_point_refinement_steps),
    )
    fixed_point_refinement_steps = int(cfg.inversion.fixed_point_refinement_steps)
    inversion_nfe = flux_inversion_nfe(
        num_inference_steps,
        solver=inversion_solver,
        fixed_point_refinement_steps=fixed_point_refinement_steps,
    )
    save_trajectories = bool(cfg.save.trajectories)
    save_velocities = bool(cfg.save.velocities)
    save_training_data = bool(cfg.save.training_data)
    editing_enabled = bool(cfg.editing.enabled)
    if editing_enabled and not target_prompt:
        raise ValueError("A non-empty target prompt is required when editing.enabled=true.")

    conditioning = encode_flux_prompt(
        pipe,
        prompt=prompt,
        max_sequence_length=int(cfg.max_sequence_length),
        device=device,
    )
    generated_noise, latent_image_ids = prepare_flux_latents(
        pipe,
        height=height,
        width=width,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    original_image_path = sample_dir / "original.png"
    sampling_trajectory: list[torch.Tensor] = []
    sampling_velocities: list[torch.Tensor] = []

    if source_sample_dir is None:
        initial_noise = generated_noise
        final_latent, schedule, sampling_trajectory, sampling_velocities = sample_flux_latent(
            pipe,
            initial_latents=initial_noise,
            conditioning=conditioning,
            latent_image_ids=latent_image_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scales.source,
            save_trajectory=save_trajectories or save_training_data,
            save_velocities=True,
            progress_desc=f"Sampling {sample_dir.name}",
        )
        original_image = decode_flux_latents(
            pipe,
            final_latent,
            height=height,
            width=width,
        )
        original_image.save(original_image_path)
        oracle_inverted_noise = oracle_invert_flux_latent(
            final_latent=final_latent.detach().cpu(),
            schedule=schedule.cpu(),
            sampling_velocities=sampling_velocities,
        )
    else:
        _validate_source_sample(
            source_sample_dir,
            expected_prompt=prompt,
            expected_seed=seed,
        )
        initial_noise_cpu = _load_tensor(source_sample_dir / "initial_noise.pt")
        _validate_source_noise_layout(
            source_sample_dir,
            initial_noise_cpu,
            generated_noise,
        )
        initial_noise = initial_noise_cpu.to(device=device)
        final_latent = _load_tensor(source_sample_dir / "final_latent.pt").to(device=device)
        if final_latent.shape != initial_noise.shape:
            raise ValueError(
                f"Source latent shapes differ in {source_sample_dir}: "
                f"initial={tuple(initial_noise.shape)}, final={tuple(final_latent.shape)}."
            )
        schedule = FluxSchedule(
            timesteps=_load_tensor(source_sample_dir / "timesteps.pt").to(device=device),
            sigmas=_load_tensor(source_sample_dir / "sigmas.pt").to(device=device),
        )
        if schedule.num_steps != num_inference_steps:
            raise ValueError(
                f"Source sample {source_sample_dir} has {schedule.num_steps} steps, "
                f"expected {num_inference_steps}."
            )
        oracle_inverted_noise = _load_tensor(
            source_sample_dir / "oracle_inverted_noise.pt"
        )
        for name in (
            "original.png",
            "initial_noise.pt",
            "final_latent.pt",
            "oracle_inverted_noise.pt",
            "timesteps.pt",
            "sigmas.pt",
        ):
            _link_source_file(source_sample_dir / name, sample_dir / name)
        source_gaussian_edit_path = source_sample_dir / "edited_from_initial_noise.png"
        source_prompt_path = source_sample_dir / "prompt.json"
        if editing_enabled and source_gaussian_edit_path.exists():
            with source_prompt_path.open("r", encoding="utf-8") as handle:
                source_metadata = json.load(handle)
            if source_metadata.get("target_prompt") == target_prompt:
                _link_source_file(
                    source_gaussian_edit_path,
                    sample_dir / "edited_from_initial_noise.png",
                )
        logger.info("{}: reusing source latent from {}", sample_dir.name, source_sample_dir)

    inversion_start_latent = nudge_flux_latent(
        final_latent,
        latent_nudging_scalar,
    )
    with _inversion_lora(pipe, enabled=bool(cfg.lora.enabled)):
        inverted_noise, inversion_trajectory, inversion_velocities = invert_flux_latent(
            pipe,
            final_latent=inversion_start_latent,
            schedule=schedule,
            conditioning=conditioning,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scales.inversion,
            solver=inversion_solver,
            fixed_point_refinement_steps=fixed_point_refinement_steps,
            save_trajectory=save_trajectories,
            save_velocities=save_velocities,
            progress_desc=f"Inverting {sample_dir.name}",
        )
    reconstructed_latent, reconstruction_schedule, _, _ = sample_flux_latent(
        pipe,
        initial_latents=inverted_noise,
        conditioning=conditioning,
        latent_image_ids=latent_image_ids,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scales.reconstruction,
        save_trajectory=False,
        save_velocities=False,
        progress_desc=f"Reconstructing {sample_dir.name}",
    )
    if not torch.equal(
        schedule.timesteps.detach().cpu(),
        reconstruction_schedule.timesteps.detach().cpu(),
    ) or not torch.equal(
        schedule.sigmas.detach().cpu(),
        reconstruction_schedule.sigmas.detach().cpu(),
    ):
        raise RuntimeError("FLUX reconstruction used a different timestep or sigma schedule.")

    reconstructed_image = decode_flux_latents(
        pipe,
        reconstructed_latent,
        height=height,
        width=width,
    )
    reconstructed_image_path = sample_dir / "reconstructed.png"
    reconstructed_image.save(reconstructed_image_path)

    edited_image_path: Path | None = None
    edited_from_initial_noise_path: Path | None = None
    if editing_enabled:
        edit_guidance_scale = guidance_scales.editing
        target_conditioning = encode_flux_prompt(
            pipe,
            prompt=str(target_prompt),
            max_sequence_length=int(cfg.max_sequence_length),
            device=device,
        )
        edited_latent, _, _, _ = sample_flux_latent(
            pipe,
            initial_latents=inverted_noise,
            conditioning=target_conditioning,
            latent_image_ids=latent_image_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=edit_guidance_scale,
            save_trajectory=False,
            save_velocities=False,
            progress_desc=f"Editing inverted {sample_dir.name}",
        )
        edited_image = decode_flux_latents(
            pipe,
            edited_latent,
            height=height,
            width=width,
        )
        edited_image_path = sample_dir / "edited.png"
        edited_image.save(edited_image_path)

        if bool(cfg.editing.generate_initial_noise_reference):
            edited_from_initial_noise_path = sample_dir / "edited_from_initial_noise.png"
            if not edited_from_initial_noise_path.exists():
                reference_edit_latent, _, _, _ = sample_flux_latent(
                    pipe,
                    initial_latents=initial_noise,
                    conditioning=target_conditioning,
                    latent_image_ids=latent_image_ids,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=edit_guidance_scale,
                    save_trajectory=False,
                    save_velocities=False,
                    progress_desc=f"Editing Gaussian {sample_dir.name}",
                )
                reference_edit_image = decode_flux_latents(
                    pipe,
                    reference_edit_latent,
                    height=height,
                    width=width,
                )
                reference_edit_image.save(edited_from_initial_noise_path)

    initial_cpu = initial_noise.detach().cpu()
    inverted_cpu = inverted_noise.detach().cpu()
    oracle_cpu = oracle_inverted_noise.detach().cpu()
    initial_unpacked = unpack_flux_latents(
        pipe,
        initial_cpu,
        height=height,
        width=width,
    )
    inverted_unpacked = unpack_flux_latents(
        pipe,
        inverted_cpu,
        height=height,
        width=width,
    )
    metrics = {
        "inversion_error": pair_metrics(initial_cpu, inverted_cpu),
        "oracle_inversion_error": pair_metrics(initial_cpu, oracle_cpu),
        "reconstruction_latent": pair_metrics(
            final_latent.detach().cpu(),
            reconstructed_latent.detach().cpu(),
        ),
        "reconstruction_latent_to_inversion_start": pair_metrics(
            inversion_start_latent.detach().cpu(),
            reconstructed_latent.detach().cpu(),
        ),
        "inversion_settings": {
            "guidance_scale": guidance_scales.inversion,
            "latent_nudging_scalar": latent_nudging_scalar,
            "solver": inversion_solver,
            "fixed_point_refinement_steps": fixed_point_refinement_steps,
            "num_steps": num_inference_steps,
            "nfe": inversion_nfe,
        },
        "reconstruction_settings": {
            "guidance_scale": guidance_scales.reconstruction,
            "num_steps": num_inference_steps,
        },
        "initial_noise_stats": tensor_stats(initial_cpu, int(cfg.metrics.max_elements)),
        "inverted_noise_stats": tensor_stats(inverted_cpu, int(cfg.metrics.max_elements)),
        "noise_normality": noise_normality_metrics(
            initial_cpu,
            inverted_cpu,
            max_elements=int(cfg.metrics.normality_sample_size),
            num_quantiles=int(cfg.metrics.qq_num_quantiles),
        ),
        "reconstruction_image": image_pair_metrics(
            original_image_path,
            reconstructed_image_path,
            plain_threshold=float(cfg.metrics.plain_threshold),
        ),
        "plain_region_noise_latent": masked_flux_noise_region_metrics(
            original_image_path,
            initial_unpacked,
            inverted_unpacked,
            plain_threshold=float(cfg.metrics.plain_threshold),
        ),
    }
    if editing_enabled:
        edit_metrics: dict[str, Any] = {
            "settings": {
                "source_prompt": prompt,
                "target_prompt": target_prompt,
                "guidance_scale": edit_guidance_scale,
            },
            "inverted_noise": {
                "image": image_pair_metrics(
                    original_image_path,
                    edited_image_path,
                    plain_threshold=float(cfg.metrics.plain_threshold),
                ),
            },
        }
        if edited_from_initial_noise_path is not None:
            edit_metrics["initial_noise"] = {
                "image": image_pair_metrics(
                    original_image_path,
                    edited_from_initial_noise_path,
                    plain_threshold=float(cfg.metrics.plain_threshold),
                ),
            }
        metrics["editing"] = edit_metrics

    if save_training_data:
        storage_dtype = resolve_torch_dtype(str(cfg.save.storage_dtype))
        if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("save.storage_dtype must be float16, bfloat16, or float32.")
        inversion_inputs, target_velocities = build_flux_inversion_pairs(
            sampling_trajectory,
            sampling_velocities,
        )
        save_flux_inversion_training_sample(
            sample_dir,
            prompt=prompt,
            sample_index=sample_index,
            seed=seed,
            guidance_scale=guidance_scales.source,
            height=height,
            width=width,
            conditioning=conditioning,
            latent_image_ids=latent_image_ids,
            schedule=schedule,
            initial_noise=initial_noise,
            final_latent=final_latent,
            inversion_inputs=inversion_inputs,
            target_velocities=target_velocities,
            storage_dtype=storage_dtype,
            model_id=str(cfg.model.model_id),
            extra_metadata={
                "dataset_split": (None if cfg.dataset_split is None else str(cfg.dataset_split)),
                "source_prompts_jsonl": (
                    None if cfg.prompts_jsonl is None else str(cfg.prompts_jsonl)
                ),
                "roundtrip_inversion_lora_enabled": bool(cfg.lora.enabled),
                "roundtrip_inversion_guidance_scale": guidance_scales.inversion,
                "roundtrip_reconstruction_guidance_scale": (
                    guidance_scales.reconstruction
                ),
                "roundtrip_latent_nudging_scalar": latent_nudging_scalar,
                "roundtrip_inversion_solver": inversion_solver,
                "roundtrip_inversion_nfe": inversion_nfe,
            },
        )

    _save_sample_tensors(
        sample_dir,
        initial_noise=initial_noise,
        final_latent=final_latent,
        inversion_start_latent=inversion_start_latent,
        inverted_noise=inverted_noise,
        oracle_inverted_noise=oracle_inverted_noise,
        reconstructed_latent=reconstructed_latent,
        schedule=schedule,
        sampling_trajectory=sampling_trajectory,
        sampling_velocities=sampling_velocities,
        inversion_trajectory=inversion_trajectory,
        inversion_velocities=inversion_velocities,
        save_trajectories=save_trajectories,
        save_velocities=save_velocities,
        save_source_inputs=source_sample_dir is None,
    )
    _save_json(
        sample_dir / "prompt.json",
        {
            "prompt": prompt,
            "target_prompt": target_prompt,
            "seed": seed,
            "latent_nudging_scalar": latent_nudging_scalar,
            "inversion_solver": inversion_solver,
            "fixed_point_refinement_steps": fixed_point_refinement_steps,
            "inversion_nfe": inversion_nfe,
            "source_guidance_scale": guidance_scales.source,
            "inversion_guidance_scale": guidance_scales.inversion,
            "reconstruction_guidance_scale": guidance_scales.reconstruction,
            "editing_guidance_scale": (
                guidance_scales.editing if editing_enabled else None
            ),
        },
    )
    _save_json(sample_dir / "metrics.json", metrics)

    logger.success(
        "{}: inversion RMSE={:.6f}, oracle RMSE={:.6f}, solver={} (NFE={}), "
        "latent nudge={:.3f}, guidance(src/inv/rec)={:.3f}/{:.3f}/{:.3f}",
        sample_dir.name,
        metrics["inversion_error"]["rmse"],
        metrics["oracle_inversion_error"]["rmse"],
        inversion_solver,
        inversion_nfe,
        latent_nudging_scalar,
        guidance_scales.source,
        guidance_scales.inversion,
        guidance_scales.reconstruction,
    )
    return metrics, initial_unpacked, inverted_unpacked


def _collect_existing_results(
    pipe: FluxPipeline,
    *,
    output_dir: Path,
    cfg: DictConfig,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    sample_results: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    initial_unpacked_samples: list[torch.Tensor] = []
    inverted_unpacked_samples: list[torch.Tensor] = []

    for sample_dir in sorted(output_dir.glob("sample_*")):
        metrics_path = sample_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        initial_noise_path = sample_dir / "initial_noise.pt"
        inverted_noise_path = sample_dir / "inverted_noise.pt"
        if not initial_noise_path.exists() or not inverted_noise_path.exists():
            sample_results[sample_dir.name] = metrics
            aggregate_rows.append(flatten_metrics(metrics))
            continue
        initial_unpacked = unpack_flux_latents(
            pipe,
            _load_tensor(initial_noise_path),
            height=int(cfg.height),
            width=int(cfg.width),
        )
        inverted_unpacked = unpack_flux_latents(
            pipe,
            _load_tensor(inverted_noise_path),
            height=int(cfg.height),
            width=int(cfg.width),
        )
        initial_unpacked_samples.append(initial_unpacked)
        inverted_unpacked_samples.append(inverted_unpacked)
        original_image_path = sample_dir / "original.png"
        if (
            "plain_region_noise_latent" not in metrics
            and original_image_path.exists()
        ):
            metrics["plain_region_noise_latent"] = masked_flux_noise_region_metrics(
                original_image_path,
                initial_unpacked,
                inverted_unpacked,
                plain_threshold=float(cfg.metrics.plain_threshold),
            )
            _save_json(metrics_path, metrics)
        sample_results[sample_dir.name] = metrics
        aggregate_rows.append(flatten_metrics(metrics))

    return (
        sample_results,
        aggregate_rows,
        initial_unpacked_samples,
        inverted_unpacked_samples,
    )


def _add_flux_clip_metrics(
    *,
    output_dir: Path,
    sample_results: dict[str, Any],
    clip_config: Any,
) -> list[dict[str, Any]]:
    if not bool(OmegaConf.select(clip_config, "enabled", default=False)):
        return []

    comparisons: list[dict[str, Any]] = []
    candidate_map: dict[str, tuple[str, str]] = {}
    for sample_name in sorted(sample_results):
        sample_dir = output_dir / sample_name
        prompt_path = sample_dir / "prompt.json"
        if not prompt_path.exists():
            continue
        with prompt_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        target_prompt = metadata.get("target_prompt")
        if not target_prompt:
            continue
        for branch, candidate_name in (
            ("initial_noise", "edited_from_initial_noise.png"),
            ("inverted_noise", "edited.png"),
        ):
            candidate_path = sample_dir / candidate_name
            if not candidate_path.exists():
                continue
            clip_sample_name = f"{sample_name}::{branch}"
            candidate_map[clip_sample_name] = (sample_name, branch)
            comparisons.append(
                {
                    "sample": clip_sample_name,
                    "source_prompt": str(metadata["prompt"]),
                    "target_prompt": str(target_prompt),
                    "final_image_path": (sample_dir / "original.png").as_posix(),
                    "candidate_image_path": candidate_path.as_posix(),
                }
            )

    temporary_results: dict[str, Any] = {}
    rows = add_clip_text_alignment_metrics(
        comparisons,
        temporary_results,
        clip_config,
    )
    for clip_sample_name, (sample_name, branch) in candidate_map.items():
        alignment = temporary_results.get(clip_sample_name, {}).get(
            "clip_text_alignment"
        )
        if not alignment:
            continue
        clip_metrics = {
            "source": alignment.get("clip_source"),
            "edit": alignment.get("clip_target"),
            "directional": alignment.get("clip_directional"),
        }
        sample_results[sample_name].setdefault("editing", {}).setdefault(
            branch, {}
        )["clip"] = clip_metrics
        _save_json(output_dir / sample_name / "metrics.json", sample_results[sample_name])
    return rows


def _batch_latent_normality_metrics(
    initial_batch: torch.Tensor,
    inverted_batch: torch.Tensor,
    *,
    patch_size: int,
    top_k: int,
) -> dict[str, float]:
    """Metrics and scaling used by Table 19 in Appendix P."""
    initial_batch = initial_batch.float()
    inverted_batch = inverted_batch.float()
    forward_kl = float(kl_div2(initial_batch, inverted_batch))
    reverse_kl = float(kl_div2(inverted_batch, initial_batch))
    global_forward_kl = float(kl_div(initial_batch, inverted_batch))
    global_reverse_kl = float(kl_div(inverted_batch, initial_batch))
    initial_corr = get_top_k_corr_in_patches(
        initial_batch,
        patch_size=patch_size,
        top_k=top_k,
    )
    inverted_corr = get_top_k_corr_in_patches(
        inverted_batch,
        patch_size=patch_size,
        top_k=top_k,
    )
    return {
        "initial_corr": float(initial_corr["mean"]),
        "initial_corr_std": float(initial_corr["std"]),
        "corr": float(inverted_corr["mean"]),
        "corr_std": float(inverted_corr["std"]),
        # The paper presents this per-location KL after multiplying it by 100.
        "kl": forward_kl * 100.0,
        "reverse_kl": reverse_kl * 100.0,
        "symmetric_kl": (forward_kl + reverse_kl) * 50.0,
        "per_location_kl": forward_kl,
        "per_location_reverse_kl": reverse_kl,
        "per_location_symmetric_kl": (forward_kl + reverse_kl) / 2.0,
        "per_location_kl_x100": forward_kl * 100.0,
        "per_location_reverse_kl_x100": reverse_kl * 100.0,
        "per_location_symmetric_kl_x100": (forward_kl + reverse_kl) * 50.0,
        "global_kl": global_forward_kl,
        "global_reverse_kl": global_reverse_kl,
        "global_symmetric_kl": (global_forward_kl + global_reverse_kl) / 2.0,
    }


def _log_flux_results_to_wandb(
    results: dict[str, Any],
    *,
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    if str(cfg.wandb.mode) == "disabled":
        logger.info("W&B logging disabled")
        return
    try:
        import wandb
    except ImportError:
        logger.warning("W&B logging skipped because wandb is not installed")
        return

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", wandb_dir.as_posix())
    try:
        run = wandb.init(
            project=str(cfg.wandb.project),
            entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            group=None if cfg.wandb.group is None else str(cfg.wandb.group),
            name=None if cfg.wandb.run_name is None else str(cfg.wandb.run_name),
            tags=(
                None
                if cfg.wandb.tags is None
                else list(OmegaConf.to_container(cfg.wandb.tags, resolve=True))
            ),
            job_type="evaluation",
            mode=str(cfg.wandb.mode),
            dir=wandb_dir.as_posix(),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        aggregate_metrics = {
            f"eval/{key}": value
            for key, value in flatten_metrics(results["aggregate"]).items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
        aggregate_metrics["eval/num_samples"] = int(results["num_samples"])
        run.log(aggregate_metrics)

        flat_samples = []
        for sample_name, metrics in results["samples"].items():
            flat_samples.append(
                {"sample": sample_name, **flatten_metrics(metrics)}
            )
        if flat_samples:
            columns = sorted({key for row in flat_samples for key in row})
            table = wandb.Table(columns=columns)
            for row in flat_samples:
                table.add_data(*[row.get(column) for column in columns])
            run.log({"eval/samples": table})

        preview_table = wandb.Table(
            columns=[
                "sample",
                "source_prompt",
                "target_prompt",
                "original",
                "reconstruction",
                "edit_from_gaussian",
                "edit_from_inversion",
            ]
        )
        max_previews = int(cfg.wandb.max_preview_samples)
        for sample_name in sorted(results["samples"])[:max_previews]:
            sample_dir = output_dir / sample_name
            prompt_path = sample_dir / "prompt.json"
            if not prompt_path.exists():
                continue
            with prompt_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            def optional_image(name: str):
                path = sample_dir / name
                return wandb.Image(path.as_posix()) if path.exists() else None

            preview_table.add_data(
                sample_name,
                metadata.get("prompt"),
                metadata.get("target_prompt"),
                optional_image("original.png"),
                optional_image("reconstructed.png"),
                optional_image("edited_from_initial_noise.png"),
                optional_image("edited.png"),
            )
        if preview_table.data:
            run.log({"eval/previews": preview_table})

        artifact = wandb.Artifact(
            name=f"{output_dir.name}-results",
            type="evaluation",
        )
        artifact.add_file((output_dir / "results.json").as_posix())
        artifact.add_file((output_dir / "run_config.yaml").as_posix())
        run.log_artifact(artifact)
        run.finish()
        logger.success("Logged FLUX evaluation to W&B")
    except Exception as exc:
        logger.warning("W&B logging failed: {}", exc)


@hydra.main(config_path="../../config", config_name="eval/flux_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("FLUX baseline config:\n{}", OmegaConf.to_yaml(cfg))
    if int(cfg.num_inference_steps) <= 0:
        raise ValueError("num_inference_steps must be positive.")
    guidance_scales = resolve_flux_guidance_scales(cfg)
    logger.info(
        "FLUX guidance scales: source={} inversion={} reconstruction={} editing={}",
        guidance_scales.source,
        guidance_scales.inversion,
        guidance_scales.reconstruction,
        guidance_scales.editing,
    )
    # Validate before loading the multi-billion-parameter pipeline.
    nudge_flux_latent(
        torch.zeros((), dtype=torch.float32),
        float(cfg.inversion.latent_nudging_scalar),
    )
    validate_flux_inversion_solver(
        str(cfg.inversion.solver),
        int(cfg.inversion.fixed_point_refinement_steps),
    )
    inline_prompts = [
        str(prompt) for prompt in OmegaConf.to_container(cfg.prompts, resolve=True) or []
    ]
    prompts = load_prompt_records(
        cfg.prompts_jsonl,
        inline_prompts,
        prompt_key=str(cfg.prompt_key),
    )
    prompt_jobs = build_flux_prompt_jobs(
        prompts,
        samples_per_prompt=int(cfg.samples_per_prompt),
        start_index=int(cfg.start_index),
        num_samples=None if cfg.num_samples is None else int(cfg.num_samples),
    )
    target_prompts: list[str] = []
    if cfg.editing.target_prompts_jsonl is not None:
        target_prompts = load_prompt_records(
            cfg.editing.target_prompts_jsonl,
            [],
            prompt_key=str(cfg.editing.target_prompt_key),
        )
    elif bool(cfg.editing.enabled) and cfg.editing.target_prompt_template is None:
        # Match the SDXL/TABA editing protocol: use the next source prompt as the edit.
        target_prompts = prompts
    edit_jobs = build_flux_edit_jobs(
        prompt_jobs,
        target_prompts=target_prompts,
        target_prompt_template=(
            None
            if cfg.editing.target_prompt_template is None
            else str(cfg.editing.target_prompt_template)
        ),
        target_prompt_offset=int(cfg.editing.target_prompt_offset),
    )
    if bool(cfg.editing.enabled) and any(
        not target_prompt for _, _, target_prompt in edit_jobs
    ):
        raise ValueError(
            "editing.enabled=true requires editing.target_prompts_jsonl or "
            "editing.target_prompt_template."
        )

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_run_config(output_dir, cfg)
    source_dir = None if cfg.source_dir is None else _resolve_path(cfg.source_dir)
    if source_dir is not None:
        if source_dir == output_dir:
            raise ValueError("source_dir and output_dir must be different directories.")
        if bool(cfg.save.training_data) or bool(cfg.save.trajectories) or bool(cfg.save.velocities):
            raise ValueError(
                "source_dir reuse requires save.training_data=false, "
                "save.trajectories=false, and save.velocities=false."
            )
        _validate_source_run_config(source_dir, cfg)
    OmegaConf.save(cfg, output_dir / "run_config.yaml")

    device = _resolve_device(cfg)
    pipe = _load_flux_pipeline(cfg, device)
    require_training_data = bool(cfg.save.training_data)

    for sample_index, prompt, target_prompt in edit_jobs:
        sample_name = f"sample_{sample_index:06d}"
        sample_dir = output_dir / sample_name
        if _sample_is_complete(
            sample_dir,
            require_training_data=require_training_data,
            require_editing=bool(cfg.editing.enabled),
            require_initial_noise_edit=(
                bool(cfg.editing.enabled)
                and bool(cfg.editing.generate_initial_noise_reference)
            ),
        ) and not bool(cfg.overwrite):
            logger.info("Skipping existing FLUX round-trip sample: {}", sample_dir)
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        _run_sample(
            pipe,
            sample_dir=sample_dir,
            sample_index=sample_index,
            prompt=prompt,
            target_prompt=target_prompt,
            seed=int(cfg.seed) + sample_index,
            cfg=cfg,
            source_sample_dir=(
                None if source_dir is None else source_dir / sample_name
            ),
        )

    (
        sample_results,
        aggregate_rows,
        initial_unpacked_samples,
        inverted_unpacked_samples,
    ) = _collect_existing_results(
        pipe,
        output_dir=output_dir,
        cfg=cfg,
    )

    clip_rows = _add_flux_clip_metrics(
        output_dir=output_dir,
        sample_results=sample_results,
        clip_config=cfg.metrics.clip,
    )
    # CLIP is calculated after generation, so rebuild flattened rows before aggregation.
    aggregate_rows = [flatten_metrics(metrics) for metrics in sample_results.values()]

    aggregate: dict[str, Any] = summarize_numeric_rows(aggregate_rows)
    min_correlation_samples = int(cfg.metrics.min_correlation_samples)
    if len(initial_unpacked_samples) >= min_correlation_samples:
        initial_batch = torch.cat(initial_unpacked_samples, dim=0)
        inverted_batch = torch.cat(inverted_unpacked_samples, dim=0)
        aggregate["latent_normality"] = _batch_latent_normality_metrics(
            initial_batch,
            inverted_batch,
            patch_size=int(cfg.metrics.patch_size),
            top_k=int(cfg.metrics.top_k),
        )
    else:
        logger.info(
            "Skipping patch correlation: found {} samples, need at least {}.",
            len(initial_unpacked_samples),
            min_correlation_samples,
        )

    results = {
        "model_id": str(cfg.model.model_id),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "inversion": OmegaConf.to_container(cfg.inversion, resolve=True),
        "reconstruction": OmegaConf.to_container(cfg.reconstruction, resolve=True),
        "editing": OmegaConf.to_container(cfg.editing, resolve=True),
        # Keep the legacy scalar for consumers that interpret it as source guidance.
        "guidance_scale": guidance_scales.source,
        "guidance_scales": {
            "source": guidance_scales.source,
            "inversion": guidance_scales.inversion,
            "reconstruction": guidance_scales.reconstruction,
            "editing": guidance_scales.editing,
        },
        "dataset_split": (None if cfg.dataset_split is None else str(cfg.dataset_split)),
        "prompt_source": (None if cfg.prompts_jsonl is None else str(cfg.prompts_jsonl)),
        "source_dir": None if source_dir is None else source_dir.as_posix(),
        "requested_samples": len(prompt_jobs),
        "num_samples": len(sample_results),
        "samples": sample_results,
        "clip_text_alignments": clip_rows,
        "aggregate": aggregate,
    }
    _save_json(output_dir / "results.json", results)
    _log_flux_results_to_wandb(results, output_dir=output_dir, cfg=cfg)
    logger.success("Saved FLUX baseline results to {}", output_dir)


if __name__ == "__main__":
    main()
