"""Generate teacher trajectories for inversion-specific FLUX LoRA training.

This module is project-original and uses the shared Diffusers-compatible sampler from
``diff_inversion.modeling.flux_sampling``. The pairing of a cleaner inversion query
``x_{i+1}`` with the frozen teacher velocity at ``x_i`` is this project's training-data
contract. See ``docs/flow_matching_provenance.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diff_inversion.modeling.flux_sampling import (
    encode_flux_prompt,
    prepare_flux_latents,
    sample_flux_latent,
)
from diff_inversion.utils import resolve_torch_dtype


def build_flux_inversion_pairs(
    sampling_trajectory: list[torch.Tensor],
    sampling_velocities: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair cleaner inversion queries with velocities from their noisy teacher states."""
    if len(sampling_trajectory) != len(sampling_velocities) + 1:
        raise ValueError(
            "A sampling trajectory must have one state more than velocities: "
            f"got {len(sampling_trajectory)} states and {len(sampling_velocities)} velocities."
        )
    if not sampling_velocities:
        raise ValueError("At least one sampling transition is required.")

    inversion_inputs = torch.stack(
        [_remove_single_batch(state) for state in sampling_trajectory[1:]],
        dim=0,
    )
    target_velocities = torch.stack(
        [_remove_single_batch(velocity) for velocity in sampling_velocities],
        dim=0,
    )
    if inversion_inputs.shape != target_velocities.shape:
        raise ValueError(
            "FLUX inversion inputs and teacher velocities must have matching shapes, got "
            f"{tuple(inversion_inputs.shape)} and {tuple(target_velocities.shape)}."
        )
    return inversion_inputs, target_velocities


def load_prompt_records(
    prompts_jsonl: str | Path | None,
    inline_prompts: list[str],
    *,
    prompt_key: str,
) -> list[str]:
    prompts = [" ".join(str(prompt).split()) for prompt in inline_prompts if str(prompt).strip()]
    if prompts_jsonl is not None:
        path = Path(to_absolute_path(str(prompts_jsonl)))
        if not path.exists():
            raise FileNotFoundError(f"FLUX prompt file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = line
                if isinstance(value, dict):
                    if prompt_key not in value:
                        raise KeyError(
                            f"Missing prompt key {prompt_key!r} at {path}:{line_number}."
                        )
                    value = value[prompt_key]
                prompt = " ".join(str(value).split())
                if prompt:
                    prompts.append(prompt)
    if not prompts:
        raise ValueError("Provide at least one inline prompt or prompts_jsonl record.")
    return prompts


def _remove_single_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _resolve_device(cfg: DictConfig) -> torch.device:
    configured = str(cfg.device).strip().lower()
    if configured == "auto":
        configured = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured)
    if bool(cfg.model.require_cuda) and device.type != "cuda":
        raise RuntimeError("FLUX data generation is configured with model.require_cuda=true.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _load_pipeline(cfg: DictConfig, device: torch.device) -> FluxPipeline:
    dtype = resolve_torch_dtype(str(cfg.model.torch_dtype))
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": bool(cfg.model.local_files_only),
    }
    if not bool(cfg.model.load_vae):
        load_kwargs["vae"] = None
    if cfg.model.revision is not None:
        load_kwargs["revision"] = str(cfg.model.revision)
    if cfg.model.variant is not None:
        load_kwargs["variant"] = str(cfg.model.variant)

    logger.info("Loading FLUX teacher pipeline: {}", cfg.model.model_id)
    pipe = FluxPipeline.from_pretrained(str(cfg.model.model_id), **load_kwargs)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    offload = str(cfg.model.offload).strip().lower()
    if offload == "model":
        if device.type != "cuda":
            raise ValueError("model.offload=model requires CUDA.")
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        if device.type != "cuda":
            raise ValueError("model.offload=sequential requires CUDA.")
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "none":
        pipe.to(device)
    else:
        raise ValueError("model.offload must be one of: none, model, sequential.")
    return pipe


def _execution_device(pipe: FluxPipeline) -> torch.device:
    return torch.device(pipe._execution_device)


def _atomic_json_save(path: Path, value: Any) -> None:
    # Array jobs share output_dir/run_config.json. PID alone is not unique
    # across nodes, so include a UUID to keep concurrent atomic writes disjoint.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _nested_get(value: dict[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _validate_existing_run_config(output_dir: Path, current: dict[str, Any]) -> None:
    path = output_dir / "run_config.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    checked_keys = (
        "dataset_split",
        "prompts_jsonl",
        "prompt_key",
        "inline_prompts",
        "samples_per_prompt",
        "seed",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "storage_dtype",
        "model.model_id",
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
            "Refusing to mix incompatible FLUX teacher data in "
            f"{output_dir}:\n{details}\nUse a new output_dir."
        )


def save_flux_inversion_training_sample(
    sample_dir: Path,
    *,
    prompt: str,
    sample_index: int,
    seed: int,
    guidance_scale: float,
    height: int,
    width: int,
    conditioning: dict[str, torch.Tensor],
    latent_image_ids: torch.Tensor,
    schedule,
    initial_noise: torch.Tensor,
    final_latent: torch.Tensor,
    inversion_inputs: torch.Tensor,
    target_velocities: torch.Tensor,
    storage_dtype: torch.dtype,
    model_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    float_conditioning = {
        key: (
            tensor.to(dtype=storage_dtype)
            if tensor.is_floating_point() and key not in {"text_ids"}
            else tensor
        )
        .detach()
        .cpu()
        for key, tensor in conditioning.items()
    }
    _atomic_torch_save(sample_dir / "conditioning.pt", float_conditioning)
    _atomic_torch_save(
        sample_dir / "latent_image_ids.pt",
        _remove_single_batch(latent_image_ids.detach().cpu()),
    )
    _atomic_torch_save(sample_dir / "timesteps.pt", schedule.timesteps.detach().float().cpu())
    _atomic_torch_save(sample_dir / "sigmas.pt", schedule.sigmas.detach().float().cpu())
    _atomic_torch_save(
        sample_dir / "inversion_inputs.pt",
        inversion_inputs.to(dtype=storage_dtype).cpu(),
    )
    _atomic_torch_save(
        sample_dir / "target_velocities.pt",
        target_velocities.to(dtype=storage_dtype).cpu(),
    )
    _atomic_torch_save(
        sample_dir / "initial_noise.pt",
        _remove_single_batch(initial_noise).to(dtype=storage_dtype).detach().cpu(),
    )
    _atomic_torch_save(
        sample_dir / "final_latent.pt",
        _remove_single_batch(final_latent).to(dtype=storage_dtype).detach().cpu(),
    )
    _atomic_json_save(sample_dir / "prompt.json", {"prompt": prompt})
    metadata = {
        "format": "flux_inversion_teacher_v1",
        "objective": "v_base(x_i,t_i) from inversion query x_{i+1}",
        "model_id": model_id,
        "sample_index": sample_index,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "num_inference_steps": int(schedule.num_steps),
        "image_sequence_length": int(inversion_inputs.shape[1]),
        "latent_channels": int(inversion_inputs.shape[2]),
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _atomic_json_save(sample_dir / "meta.json", metadata)


@hydra.main(config_path="../../config/data", config_name="flux_inversion", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("Expected the resolved FLUX data config to be a mapping.")
    _validate_existing_run_config(output_dir, resolved_config)
    _atomic_json_save(
        output_dir / "run_config.json",
        resolved_config,
    )

    inline_prompts = [
        str(value) for value in OmegaConf.to_container(cfg.inline_prompts, resolve=True) or []
    ]
    prompts = load_prompt_records(
        cfg.prompts_jsonl,
        inline_prompts,
        prompt_key=str(cfg.prompt_key),
    )
    samples_per_prompt = int(cfg.samples_per_prompt)
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive.")
    jobs = [prompt for prompt in prompts for _ in range(samples_per_prompt)]

    start_index = int(cfg.start_index)
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    available = len(jobs) - start_index
    num_samples = available if cfg.num_samples is None else int(cfg.num_samples)
    if num_samples <= 0 or num_samples > available:
        raise ValueError(
            f"Requested {num_samples} samples from start_index={start_index}, "
            f"but only {max(available, 0)} jobs are available."
        )

    device = _resolve_device(cfg)
    storage_dtype = resolve_torch_dtype(str(cfg.storage_dtype))
    if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("storage_dtype must be float16, bfloat16, or float32.")
    pipe = _load_pipeline(cfg, device)
    execution_device = _execution_device(pipe)

    generated = 0
    skipped = 0
    for sample_index in tqdm(
        range(start_index, start_index + num_samples),
        desc="FLUX teacher trajectories",
    ):
        sample_dir = output_dir / f"sample_{sample_index:06d}"
        if (sample_dir / "meta.json").exists() and not bool(cfg.overwrite):
            skipped += 1
            continue

        prompt = jobs[sample_index]
        seed = int(cfg.seed) + sample_index
        conditioning = encode_flux_prompt(
            pipe,
            prompt=prompt,
            max_sequence_length=int(cfg.max_sequence_length),
            device=execution_device,
        )
        initial_noise, latent_image_ids = prepare_flux_latents(
            pipe,
            height=int(cfg.height),
            width=int(cfg.width),
            generator=torch.Generator(device=execution_device).manual_seed(seed),
            device=execution_device,
        )
        final_latent, schedule, trajectory, velocities = sample_flux_latent(
            pipe,
            initial_latents=initial_noise,
            conditioning=conditioning,
            latent_image_ids=latent_image_ids,
            num_inference_steps=int(cfg.num_inference_steps),
            guidance_scale=float(cfg.guidance_scale),
            save_trajectory=True,
            save_velocities=True,
            progress_desc=f"sample {sample_index}",
        )
        inversion_inputs, target_velocities = build_flux_inversion_pairs(
            trajectory,
            velocities,
        )
        save_flux_inversion_training_sample(
            sample_dir,
            prompt=prompt,
            sample_index=sample_index,
            seed=seed,
            guidance_scale=float(cfg.guidance_scale),
            height=int(cfg.height),
            width=int(cfg.width),
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
                "dataset_split": str(cfg.dataset_split),
                "source_prompts_jsonl": (
                    None if cfg.prompts_jsonl is None else str(cfg.prompts_jsonl)
                ),
            },
        )
        generated += 1

    logger.success(
        "FLUX training data ready in {} (generated={}, skipped={})",
        output_dir,
        generated,
        skipped,
    )


if __name__ == "__main__":
    main()
