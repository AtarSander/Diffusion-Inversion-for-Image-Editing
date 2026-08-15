"""Train a FLUX LoRA to correct velocity predictions used during ODE inversion.

This is a project-original trainer built from the public Accelerate, Diffusers, and PEFT
APIs; it is not a vendored external training script. The inversion-only teacher/student
objective, balanced transition sampling, validation, and resumable checkpoint contract
are project code. See ``docs/flow_matching_provenance.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
import socket
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, gather_object, set_seed
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diff_inversion.data.flux_inversion_dataset import (
    BalancedInversionStepSampler,
    FluxInversionDataset,
    collate_flux_inversion_batch,
)
from diff_inversion.utils import resolve_torch_dtype


def _configure_slurm_distributed_environment() -> None:
    """Map Slurm task variables to the torchrun variables Accelerate expects."""
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if world_size <= 1 or "LOCAL_RANK" in os.environ:
        return
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    tasks_per_node = str(os.environ.get("SLURM_NTASKS_PER_NODE", world_size))
    local_world_size = int(tasks_per_node.split("(", maxsplit=1)[0])
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(local_world_size))
    os.environ.setdefault("MASTER_ADDR", socket.gethostname())
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    os.environ.setdefault("MASTER_PORT", str(29500 + job_id % 1000))


def predict_flux_inversion_velocity(
    transformer: FluxTransformer2DModel,
    batch: dict[str, torch.Tensor],
    *,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the student at the cleaner point available to explicit inversion."""
    inversion_input = batch["inversion_input"].to(dtype=weight_dtype)
    timestep = batch["timestep"].flatten().to(dtype=weight_dtype) / 1000
    prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)
    guidance = batch["guidance_scale"].flatten().to(dtype=torch.float32)

    return transformer(
        hidden_states=inversion_input,
        timestep=timestep,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=batch["text_ids"],
        img_ids=batch["latent_image_ids"],
        joint_attention_kwargs={},
        return_dict=False,
    )[0]


def flux_inversion_velocity_loss(
    transformer: FluxTransformer2DModel,
    batch: dict[str, torch.Tensor],
    *,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = predict_flux_inversion_velocity(
        transformer,
        batch,
        weight_dtype=weight_dtype,
    )
    target = batch["target_velocity"].to(dtype=weight_dtype)
    error = prediction.float() - target.float()
    mse_per_item = error.square().flatten(1).mean(1)
    loss = mse_per_item.mean()
    return loss, {
        "loss": loss.detach(),
        "mse": loss.detach(),
        "mse_per_item": mse_per_item.detach(),
        "mae": error.abs().mean().detach(),
        "target_rms": target.float().square().mean().sqrt().detach(),
        "prediction_rms": prediction.float().square().mean().sqrt().detach(),
    }


def _accumulate_per_step_mse(
    totals: torch.Tensor,
    counts: torch.Tensor,
    inversion_steps: torch.Tensor,
    mse_per_item: torch.Tensor,
) -> None:
    inversion_steps = inversion_steps.flatten().to(device=totals.device, dtype=torch.long)
    mse_per_item = mse_per_item.flatten().to(device=totals.device, dtype=totals.dtype)
    if inversion_steps.shape != mse_per_item.shape:
        raise ValueError(
            "inversion_steps and mse_per_item must contain the same number of values."
        )
    if bool(((inversion_steps < 0) | (inversion_steps >= totals.numel())).any()):
        raise ValueError(
            f"Training batch contains an inversion_step outside [0, {totals.numel() - 1}]."
        )
    totals.index_add_(0, inversion_steps, mse_per_item)
    counts.index_add_(
        0,
        inversion_steps,
        torch.ones_like(mse_per_item, dtype=counts.dtype, device=counts.device),
    )


def _format_per_step_mse(
    totals: torch.Tensor,
    counts: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for inversion_step in range(totals.numel()):
        count = float(counts[inversion_step].cpu())
        if count == 0:
            continue
        result[f"{prefix}/inversion_step_{inversion_step:02d}/mse"] = float(
            (totals[inversion_step] / counts[inversion_step]).cpu()
        )
    return result


def _atomic_json_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _resolve_training_state_path(path: str | Path) -> Path:
    resolved = Path(to_absolute_path(str(path)))
    if resolved.is_dir():
        resolved = resolved / "training_state.pt"
    if not resolved.is_file():
        raise FileNotFoundError(f"FLUX training-state checkpoint not found: {resolved}")
    return resolved


def _load_training_state(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_training_state_path(path)
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    required = {
        "global_step",
        "lora_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "torch_rng_state",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        missing = required.difference(state if isinstance(state, dict) else {})
        raise ValueError(
            f"Not a complete FLUX training-state checkpoint: {resolved}; "
            f"missing {sorted(missing)}."
        )
    state["_path"] = resolved
    return state


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng_state"])
    cuda_state = state.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _wandb_mode(cfg: DictConfig) -> str:
    wandb_cfg = cfg.get("wandb")
    mode = "disabled" if wandb_cfg is None else str(wandb_cfg.get("mode", "disabled"))
    mode = mode.strip().lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be one of: online, offline, disabled.")
    return mode


def _wandb_init_kwargs(cfg: DictConfig, output_dir: Path) -> dict[str, Any]:
    wandb_cfg = cfg.wandb
    wandb_root = Path(os.environ.get("WANDB_DIR", output_dir / "wandb"))
    cache_dir = Path(os.environ.get("WANDB_CACHE_DIR", wandb_root / "cache"))
    config_dir = Path(os.environ.get("WANDB_CONFIG_DIR", wandb_root / "config"))
    data_dir = Path(os.environ.get("WANDB_DATA_DIR", wandb_root / "data"))
    for directory in (wandb_root, cache_dir, config_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", wandb_root.as_posix())
    os.environ.setdefault("WANDB_CACHE_DIR", cache_dir.as_posix())
    os.environ.setdefault("WANDB_CONFIG_DIR", config_dir.as_posix())
    os.environ.setdefault("WANDB_DATA_DIR", data_dir.as_posix())

    tags = OmegaConf.to_container(wandb_cfg.get("tags", []), resolve=True) or []
    values: dict[str, Any] = {
        "entity": wandb_cfg.get("entity"),
        "group": wandb_cfg.get("group"),
        "name": wandb_cfg.get("run_name"),
        "job_type": wandb_cfg.get("job_type", "train"),
        "mode": _wandb_mode(cfg),
        "id": wandb_cfg.get("id"),
        "resume": wandb_cfg.get("resume"),
        "tags": [str(tag) for tag in tags],
        "dir": wandb_root.as_posix(),
    }
    return {key: value for key, value in values.items() if value is not None and value != ""}


def _initialize_trackers(
    accelerator: Accelerator,
    cfg: DictConfig,
    output_dir: Path,
    resolved_config: dict[str, Any],
) -> bool:
    if _wandb_mode(cfg) == "disabled":
        logger.info("W&B logging disabled")
        return False

    project = str(cfg.wandb.get("project", "diff-inversion"))
    accelerator.init_trackers(
        project,
        config=resolved_config,
        init_kwargs={"wandb": _wandb_init_kwargs(cfg, output_dir)},
    )
    if accelerator.is_main_process:
        run = accelerator.get_tracker("wandb", unwrap=True)
        metadata = {
            "entity": getattr(run, "entity", None),
            "project": getattr(run, "project", project),
            "id": getattr(run, "id", None),
            "name": getattr(run, "name", None),
            "group": getattr(run, "group", None),
            "url": getattr(run, "url", None),
            "mode": _wandb_mode(cfg),
        }
        _atomic_json_save(output_dir / "wandb_run.json", metadata)
        logger.success("W&B run: {}", metadata["url"] or metadata["id"])
    return True


def _log_tracker_metrics(
    accelerator: Accelerator,
    values: dict[str, Any],
    *,
    step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    metrics = {key: value for key, value in values.items() if key != "step"}
    metrics["trainer/global_step"] = step
    accelerator.log(metrics, step=step)


def _dataset(
    root_dir: str | Path,
    cfg: DictConfig,
    *,
    max_samples: int | None = None,
    sample_seed: int = 0,
) -> FluxInversionDataset:
    max_step = cfg.max_inversion_step
    return FluxInversionDataset(
        Path(to_absolute_path(str(root_dir))),
        min_inversion_step=int(cfg.min_inversion_step),
        max_inversion_step=None if max_step is None else int(max_step),
        max_samples=max_samples,
        sample_seed=sample_seed,
    )


def _dataloader(
    dataset: FluxInversionDataset,
    cfg: DictConfig,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    balance_inversion_steps: bool = False,
    seed: int = 0,
) -> DataLoader:
    sampler = (
        BalancedInversionStepSampler(dataset, seed=seed)
        if balance_inversion_steps
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size if batch_size is None else batch_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.num_workers) > 0,
        collate_fn=collate_flux_inversion_batch,
    )


def _validate_dataset_compatibility(
    dataset: FluxInversionDataset,
    *,
    expected_model_id: str,
    expected_guidance_scale: float | None,
    expected_num_inference_steps: int | None,
) -> int:
    mismatches = sorted(
        {
            str(sample["meta"]["model_id"])
            for sample in dataset.samples
            if sample["meta"].get("model_id") is not None
            and str(sample["meta"]["model_id"]) != expected_model_id
        }
    )
    if mismatches:
        raise ValueError(
            f"Dataset teacher model(s) {mismatches} do not match training base "
            f"model {expected_model_id!r}."
        )

    guidance_scales = sorted(
        {
            float(sample["meta"]["guidance_scale"])
            for sample in dataset.samples
            if sample["meta"].get("guidance_scale") is not None
        }
    )
    if len(guidance_scales) != 1:
        raise ValueError(
            "Every FLUX teacher sample in a split must use the same embedded "
            f"guidance scale; found {guidance_scales}."
        )
    if expected_guidance_scale is not None and not math.isclose(
        guidance_scales[0],
        expected_guidance_scale,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Dataset embedded guidance is {guidance_scales[0]}, but training expects "
            f"{expected_guidance_scale}. Regenerate the teacher pairs or change "
            "data.expected_guidance_scale explicitly."
        )

    step_counts = sorted({int(sample["num_steps"]) for sample in dataset.samples})
    if len(step_counts) != 1:
        raise ValueError(
            "Every FLUX teacher sample in a split must use the same number of "
            f"inference steps; found {step_counts}."
        )
    if expected_num_inference_steps is not None and step_counts[0] != expected_num_inference_steps:
        raise ValueError(
            f"Dataset contains {step_counts[0]} steps, but training expects "
            f"{expected_num_inference_steps}."
        )
    return step_counts[0]


def _load_transformer(cfg: DictConfig, weight_dtype: torch.dtype) -> FluxTransformer2DModel:
    load_kwargs: dict[str, Any] = {
        "subfolder": "transformer",
        "torch_dtype": weight_dtype,
        "local_files_only": bool(cfg.local_files_only),
    }
    if cfg.revision is not None:
        load_kwargs["revision"] = str(cfg.revision)
    if cfg.variant is not None:
        load_kwargs["variant"] = str(cfg.variant)
    return FluxTransformer2DModel.from_pretrained(str(cfg.model_id), **load_kwargs)


def _install_lora(transformer: FluxTransformer2DModel, cfg: DictConfig) -> list[torch.Tensor]:
    transformer.requires_grad_(False)
    target_modules = [
        str(value) for value in OmegaConf.to_container(cfg.target_modules, resolve=True) or []
    ]
    if not target_modules:
        raise ValueError("lora.target_modules cannot be empty.")
    transformer.add_adapter(
        LoraConfig(
            r=int(cfg.r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            init_lora_weights=cfg.init_lora_weights,
            target_modules=target_modules,
            bias="none",
        )
    )
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were installed.")
    for parameter in trainable:
        parameter.data = parameter.data.float()
    return trainable


def _save_lora(
    accelerator: Accelerator,
    transformer: FluxTransformer2DModel,
    output_dir: Path,
    *,
    global_step: int,
    cfg: DictConfig,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(transformer)
    state_dict = get_peft_model_state_dict(unwrapped)
    state_dict = convert_state_dict_to_diffusers(state_dict)
    peft_configs = getattr(unwrapped, "peft_config", {})
    if len(peft_configs) != 1:
        raise RuntimeError(
            "Expected exactly one FLUX LoRA adapter when saving, found "
            f"{len(peft_configs)}."
        )
    adapter_metadata = next(iter(peft_configs.values())).to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    FluxPipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=state_dict,
        transformer_lora_adapter_metadata=adapter_metadata,
        safe_serialization=True,
    )
    _atomic_json_save(
        output_dir / "training_metadata.json",
        {
            "base_model": str(cfg.model.model_id),
            "global_step": global_step,
            "objective": "teacher velocity v_base(x_i,t_i) predicted from x_{i+1}",
            "embedded_guidance_scale": cfg.data.expected_guidance_scale,
            "num_inference_steps": cfg.data.expected_num_inference_steps,
            "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        },
    )
    logger.success("Saved FLUX inversion LoRA to {}", output_dir)


def _save_training_checkpoint(
    accelerator: Accelerator,
    transformer: FluxTransformer2DModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    global_step: int,
    data_epoch: int,
    batches_consumed_in_epoch: int,
    cfg: DictConfig,
) -> None:
    _save_lora(
        accelerator,
        transformer,
        output_dir,
        global_step=global_step,
        cfg=cfg,
    )
    accelerator.wait_for_everyone()
    if not bool(cfg.save_training_state):
        return

    rng_states = gather_object([_capture_rng_state()])
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(transformer)
    state = {
        "format_version": 1,
        "global_step": int(global_step),
        "data_epoch": int(data_epoch),
        "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "batch_size": int(cfg.data.batch_size),
        "lora_state_dict": {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(unwrapped).items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "rng_states_by_process": rng_states,
        **rng_states[0],
    }
    state_path = output_dir / "training_state.pt"
    _atomic_torch_save(state_path, state)
    logger.success("Saved resumable FLUX training state to {}", state_path)


@torch.no_grad()
def _validate(
    transformer: FluxTransformer2DModel,
    loader: DataLoader | None,
    *,
    weight_dtype: torch.dtype,
    accelerator: Accelerator,
    max_batches: int | None,
    num_inference_steps: int,
    report_per_step: bool,
) -> dict[str, float]:
    if loader is None:
        return {}
    transformer.eval()
    totals = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    step_totals = torch.zeros(
        (num_inference_steps, 2),
        device=accelerator.device,
        dtype=torch.float64,
    )
    step_counts = torch.zeros(
        num_inference_steps,
        device=accelerator.device,
        dtype=torch.float64,
    )
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prediction = predict_flux_inversion_velocity(
            transformer,
            batch,
            weight_dtype=weight_dtype,
        )
        target = batch["target_velocity"].to(dtype=weight_dtype)
        error = prediction.float() - target.float()
        batch_size = int(batch["inversion_input"].shape[0])
        per_item = torch.stack(
            [
                error.square().flatten(1).mean(1),
                error.abs().flatten(1).mean(1),
                target.float().square().flatten(1).mean(1).sqrt(),
                prediction.float().square().flatten(1).mean(1).sqrt(),
            ],
            dim=1,
        ).to(dtype=torch.float64)
        totals += per_item.sum(dim=0)
        count += batch_size

        inversion_steps = batch["inversion_step"].flatten().to(dtype=torch.long)
        if bool(((inversion_steps < 0) | (inversion_steps >= num_inference_steps)).any()):
            raise ValueError(
                "Validation batch contains an inversion_step outside the configured "
                f"range [0, {num_inference_steps - 1}]."
            )
        step_totals.index_add_(0, inversion_steps, per_item[:, :2])
        step_counts.index_add_(
            0,
            inversion_steps,
            torch.ones(batch_size, device=accelerator.device, dtype=torch.float64),
        )

    totals = accelerator.reduce(totals, reduction="sum")
    step_totals = accelerator.reduce(step_totals, reduction="sum")
    step_counts = accelerator.reduce(step_counts, reduction="sum")
    count_tensor = accelerator.reduce(
        torch.tensor(count, device=accelerator.device, dtype=torch.float64),
        reduction="sum",
    )
    transformer.train()
    if count_tensor.item() == 0:
        return {}
    values = totals / count_tensor
    result = {
        "val/loss": float(values[0].cpu()),
        "val/mse": float(values[0].cpu()),
        "val/mae": float(values[1].cpu()),
        "val/target_rms": float(values[2].cpu()),
        "val/prediction_rms": float(values[3].cpu()),
    }
    if report_per_step:
        for inversion_step in range(num_inference_steps):
            step_count = float(step_counts[inversion_step].cpu())
            if step_count == 0:
                continue
            step_values = step_totals[inversion_step] / step_counts[inversion_step]
            prefix = f"val/inversion_step_{inversion_step:02d}"
            result[f"{prefix}/mse"] = float(step_values[0].cpu())
            result[f"{prefix}/mae"] = float(step_values[1].cpu())
    return result


@hydra.main(config_path="../../config", config_name="train_flux_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    _configure_slurm_distributed_environment()
    if int(cfg.max_train_steps) <= 0:
        raise ValueError("max_train_steps must be positive.")
    if int(cfg.log_every_steps) <= 0:
        raise ValueError("log_every_steps must be positive.")
    if int(cfg.train_per_step_log_every_steps) <= 0:
        raise ValueError("train_per_step_log_every_steps must be positive.")
    if cfg.max_validation_batches is not None and int(cfg.max_validation_batches) <= 0:
        raise ValueError("max_validation_batches must be positive or null.")
    if int(cfg.data.batch_size) <= 0 or int(cfg.data.validation_batch_size) <= 0:
        raise ValueError("Training and validation batch sizes must be positive.")
    if (
        cfg.data.validation_num_trajectories is not None
        and int(cfg.data.validation_num_trajectories) <= 0
    ):
        raise ValueError("data.validation_num_trajectories must be positive or null.")
    if bool(cfg.model.require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("FLUX LoRA training is configured with model.require_cuda=true.")

    resume_enabled = bool(cfg.resume.enabled)
    resume_state = None
    if resume_enabled:
        if not cfg.resume.checkpoint_path:
            raise ValueError(
                "resume.checkpoint_path must be set when resume.enabled=true."
            )
        resume_state = _load_training_state(str(cfg.resume.checkpoint_path))

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    run_config_path = output_dir / "run_config.json"
    existing_markers = (
        run_config_path,
        metrics_path,
        output_dir / "final",
    )
    if (
        any(path.exists() for path in existing_markers)
        and not bool(cfg.overwrite)
        and not resume_enabled
    ):
        raise FileExistsError(
            f"FLUX LoRA output already contains a run: {output_dir}. "
            "Use a new output_dir or set overwrite=true explicitly."
        )
    if metrics_path.exists() and bool(cfg.overwrite) and not resume_enabled:
        metrics_path.unlink()

    tracking_enabled = _wandb_mode(cfg) != "disabled"
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=int(cfg.gradient_accumulation_steps),
            # Keep accumulation continuous across DataLoader epochs. Otherwise
            # Accelerate forces a short optimizer batch at every epoch end.
            sync_with_dataloader=False,
        ),
        mixed_precision=str(cfg.mixed_precision),
        log_with="wandb" if tracking_enabled else None,
    )
    if bool(cfg.model.require_cuda) and accelerator.device.type != "cuda":
        raise RuntimeError("FLUX LoRA training is configured with model.require_cuda=true.")
    set_seed(int(cfg.seed), device_specific=True)

    weight_dtype = resolve_torch_dtype(str(cfg.model.torch_dtype))
    if weight_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("model.torch_dtype must be float16, bfloat16, or float32.")

    train_dataset = _dataset(cfg.data.train_dir, cfg.data)
    val_dataset = (
        None
        if cfg.data.val_dir is None
        else _dataset(
            cfg.data.val_dir,
            cfg.data,
            max_samples=(
                None
                if cfg.data.validation_num_trajectories is None
                else int(cfg.data.validation_num_trajectories)
            ),
            sample_seed=int(cfg.data.validation_sample_seed),
        )
    )
    expected_model_id = str(cfg.model.model_id)
    expected_guidance_scale = (
        None
        if cfg.data.expected_guidance_scale is None
        else float(cfg.data.expected_guidance_scale)
    )
    expected_num_inference_steps = (
        None
        if cfg.data.expected_num_inference_steps is None
        else int(cfg.data.expected_num_inference_steps)
    )
    train_num_inference_steps = _validate_dataset_compatibility(
        train_dataset,
        expected_model_id=expected_model_id,
        expected_guidance_scale=expected_guidance_scale,
        expected_num_inference_steps=expected_num_inference_steps,
    )
    if val_dataset is not None:
        val_num_inference_steps = _validate_dataset_compatibility(
            val_dataset,
            expected_model_id=expected_model_id,
            expected_guidance_scale=expected_guidance_scale,
            expected_num_inference_steps=expected_num_inference_steps,
        )
        if val_num_inference_steps != train_num_inference_steps:
            raise ValueError(
                "Training and validation trajectories use different step counts: "
                f"{train_num_inference_steps} and {val_num_inference_steps}."
            )
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("Expected the resolved FLUX training config to be a mapping.")
    if accelerator.is_main_process:
        _atomic_json_save(run_config_path, resolved_config)
    accelerator.wait_for_everyone()
    tracking_enabled = _initialize_trackers(
        accelerator,
        cfg,
        output_dir,
        resolved_config,
    )
    train_loader = _dataloader(
        train_dataset,
        cfg.data,
        shuffle=not bool(cfg.data.balance_inversion_steps),
        balance_inversion_steps=bool(cfg.data.balance_inversion_steps),
        seed=int(cfg.seed),
    )
    train_sampler = (
        train_loader.sampler
        if isinstance(train_loader.sampler, BalancedInversionStepSampler)
        else None
    )
    val_loader = (
        None
        if val_dataset is None
        else _dataloader(
            val_dataset,
            cfg.data,
            shuffle=False,
            batch_size=int(cfg.data.validation_batch_size),
        )
    )

    transformer = _load_transformer(cfg.model, weight_dtype)
    trainable_parameters = _install_lora(transformer, cfg.lora)
    if resume_state is not None:
        set_peft_model_state_dict(transformer, resume_state["lora_state_dict"])
    if bool(cfg.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.learning_rate),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        weight_decay=float(cfg.weight_decay),
        eps=float(cfg.adam_epsilon),
    )
    # Accelerate advances a prepared scheduler once per process when each DDP
    # process receives a distinct batch. Scale the underlying schedule so its
    # warmup and decay still span cfg.max_train_steps global optimizer updates.
    scheduler_process_multiplier = max(1, int(accelerator.num_processes))
    lr_scheduler = get_scheduler(
        str(cfg.lr_scheduler.name),
        optimizer=optimizer,
        num_warmup_steps=(
            int(cfg.lr_scheduler.warmup_steps) * scheduler_process_multiplier
        ),
        num_training_steps=int(cfg.max_train_steps) * scheduler_process_multiplier,
        num_cycles=float(cfg.lr_scheduler.num_cycles),
        power=float(cfg.lr_scheduler.power),
    )
    if resume_state is not None:
        stored_gradient_accumulation = int(
            resume_state.get(
                "gradient_accumulation_steps",
                cfg.gradient_accumulation_steps,
            )
        )
        stored_batch_size = int(resume_state.get("batch_size", cfg.data.batch_size))
        if stored_gradient_accumulation != int(cfg.gradient_accumulation_steps):
            raise ValueError(
                "Exact resume requires the same gradient_accumulation_steps: "
                f"checkpoint={stored_gradient_accumulation}, "
                f"config={cfg.gradient_accumulation_steps}."
            )
        if stored_batch_size != int(cfg.data.batch_size):
            raise ValueError(
                "Exact resume requires the same physical batch size: "
                f"checkpoint={stored_batch_size}, config={cfg.data.batch_size}."
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(resume_state["lr_scheduler_state_dict"])

    if val_loader is None:
        transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            transformer,
            optimizer,
            train_loader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            transformer,
            optimizer,
            train_loader,
            val_loader,
            lr_scheduler,
        )

    global_step = 0 if resume_state is None else int(resume_state["global_step"])
    data_epoch = 0 if resume_state is None else int(resume_state.get("data_epoch", 0))
    batches_consumed_in_epoch = (
        0
        if resume_state is None
        else int(resume_state.get("batches_consumed_in_epoch", 0))
    )
    if global_step >= int(cfg.max_train_steps):
        raise ValueError(
            "Resume step must be below max_train_steps: "
            f"resume={global_step}, max_train_steps={cfg.max_train_steps}."
    )
    if resume_state is not None:
        rng_states = resume_state.get("rng_states_by_process")
        if rng_states is not None:
            if len(rng_states) != accelerator.num_processes:
                raise ValueError(
                    "Exact resume requires the same number of distributed processes: "
                    f"checkpoint={len(rng_states)}, current={accelerator.num_processes}."
                )
            _restore_rng_state(rng_states[accelerator.process_index])
        else:
            _restore_rng_state(resume_state)
        if hasattr(train_loader, "iteration"):
            # DataLoaderShard calls sampler.set_epoch(self.iteration) when an
            # iterator is created, so align that internal counter as well.
            train_loader.iteration = data_epoch
        logger.success(
            "Resumed FLUX training from {} at optimizer step {}, data epoch {}, batch {}",
            resume_state["_path"],
            global_step,
            data_epoch,
            batches_consumed_in_epoch,
        )

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info(
        "FLUX LoRA training: samples={} items={} val_samples={} val_items={} "
        "steps={} guidance={} trainable={:,} device={} dtype={}",
        len(train_dataset.samples),
        len(train_dataset),
        0 if val_dataset is None else len(val_dataset.samples),
        0 if val_dataset is None else len(val_dataset),
        train_num_inference_steps,
        expected_guidance_scale,
        trainable_count,
        accelerator.device,
        weight_dtype,
    )

    if val_loader is not None and bool(cfg.validate_before_training) and global_step == 0:
        baseline_validation = _validate(
            transformer,
            val_loader,
            weight_dtype=weight_dtype,
            accelerator=accelerator,
            max_batches=(
                None if cfg.max_validation_batches is None else int(cfg.max_validation_batches)
            ),
            num_inference_steps=train_num_inference_steps,
            report_per_step=bool(cfg.report_validation_per_step),
        )
        baseline_validation["step"] = 0
        if accelerator.is_main_process:
            _append_jsonl(metrics_path, baseline_validation)
            logger.info(
                "step=0 baseline val/mse={:.6g}",
                baseline_validation["val/mse"],
            )
        _log_tracker_metrics(
            accelerator,
            baseline_validation,
            step=0,
            enabled=tracking_enabled,
        )

    scalar_metric_names = ("loss", "mse", "mae", "target_rms", "prediction_rms")
    recent_metric_totals = {name: 0.0 for name in scalar_metric_names}
    recent_micro_steps = 0
    train_step_mse_totals = torch.zeros(
        train_num_inference_steps,
        device=accelerator.device,
        dtype=torch.float64,
    )
    train_step_counts = torch.zeros_like(train_step_mse_totals)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=int(cfg.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="FLUX inversion LoRA",
    )
    transformer.train()

    while global_step < int(cfg.max_train_steps):
        if train_sampler is not None:
            train_sampler.set_epoch(data_epoch)
        skip_batches = batches_consumed_in_epoch
        for batch_index, batch in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            batches_consumed_in_epoch = batch_index + 1
            with accelerator.accumulate(transformer):
                loss, metrics = flux_inversion_velocity_loss(
                    transformer,
                    batch,
                    weight_dtype=weight_dtype,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite FLUX LoRA loss at global_step={global_step}: {loss.item()}."
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        float(cfg.max_grad_norm),
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            for metric_name in scalar_metric_names:
                recent_metric_totals[metric_name] += float(
                    metrics[metric_name].detach().float().cpu()
                )
            recent_micro_steps += 1
            if bool(cfg.report_train_per_step):
                _accumulate_per_step_mse(
                    train_step_mse_totals,
                    train_step_counts,
                    batch["inversion_step"],
                    metrics["mse_per_item"],
                )
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{float(metrics['mse'].cpu()):.4g}")

            row: dict[str, Any] = {}
            if global_step % int(cfg.log_every_steps) == 0:
                local_metric_window = torch.tensor(
                    [
                        *(recent_metric_totals[name] for name in scalar_metric_names),
                        recent_micro_steps,
                    ],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                reduced_metric_window = accelerator.reduce(
                    local_metric_window,
                    reduction="sum",
                )
                reduced_micro_steps = max(float(reduced_metric_window[-1].cpu()), 1.0)
                row.update(
                    {
                        f"train/{name}": float(reduced_metric_window[index].cpu())
                        / reduced_micro_steps
                        for index, name in enumerate(scalar_metric_names)
                    }
                )
                row["train/lr"] = float(lr_scheduler.get_last_lr()[0])
                row["train/physical_batch_size"] = int(cfg.data.batch_size)
                row["train/effective_batch_size"] = int(cfg.data.batch_size) * int(
                    cfg.gradient_accumulation_steps
                ) * int(accelerator.num_processes)
                if accelerator.device.type == "cuda":
                    row["train/gpu_peak_allocated_gib"] = float(
                        torch.cuda.max_memory_allocated(accelerator.device) / 2**30
                    )
                    row["train/gpu_peak_reserved_gib"] = float(
                        torch.cuda.max_memory_reserved(accelerator.device) / 2**30
                    )
                recent_metric_totals = {name: 0.0 for name in scalar_metric_names}
                recent_micro_steps = 0

            if (
                bool(cfg.report_train_per_step)
                and global_step % int(cfg.train_per_step_log_every_steps) == 0
            ):
                reduced_step_totals = accelerator.reduce(
                    train_step_mse_totals,
                    reduction="sum",
                )
                reduced_step_counts = accelerator.reduce(
                    train_step_counts,
                    reduction="sum",
                )
                row.update(
                    _format_per_step_mse(
                        reduced_step_totals,
                        reduced_step_counts,
                        prefix="train",
                    )
                )
                train_step_mse_totals.zero_()
                train_step_counts.zero_()

            if row:
                row["step"] = global_step
                if accelerator.is_main_process:
                    _append_jsonl(metrics_path, row)
                    if "train/loss" in row:
                        logger.info(
                            "step={} train/loss={:.6g} train/mse={:.6g}",
                            global_step,
                            row["train/loss"],
                            row["train/mse"],
                        )
                _log_tracker_metrics(
                    accelerator,
                    row,
                    step=global_step,
                    enabled=tracking_enabled,
                )

            if (
                val_loader is not None
                and int(cfg.validate_every_steps) > 0
                and global_step % int(cfg.validate_every_steps) == 0
            ):
                validation = _validate(
                    transformer,
                    val_loader,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    max_batches=(
                        None
                        if cfg.max_validation_batches is None
                        else int(cfg.max_validation_batches)
                    ),
                    num_inference_steps=train_num_inference_steps,
                    report_per_step=bool(cfg.report_validation_per_step),
                )
                validation["step"] = global_step
                if accelerator.is_main_process:
                    _append_jsonl(metrics_path, validation)
                    logger.info("step={} val/mse={:.6g}", global_step, validation["val/mse"])
                _log_tracker_metrics(
                    accelerator,
                    validation,
                    step=global_step,
                    enabled=tracking_enabled,
                )

            if int(cfg.save_every_steps) > 0 and global_step % int(cfg.save_every_steps) == 0:
                _save_training_checkpoint(
                    accelerator,
                    transformer,
                    optimizer,
                    lr_scheduler,
                    output_dir / f"checkpoint-{global_step}",
                    global_step=global_step,
                    data_epoch=data_epoch,
                    batches_consumed_in_epoch=batches_consumed_in_epoch,
                    cfg=cfg,
                )

            if global_step >= int(cfg.max_train_steps):
                break

        if global_step < int(cfg.max_train_steps):
            data_epoch += 1
            batches_consumed_in_epoch = 0

    progress.close()
    _save_training_checkpoint(
        accelerator,
        transformer,
        optimizer,
        lr_scheduler,
        output_dir / "final",
        global_step=global_step,
        data_epoch=data_epoch,
        batches_consumed_in_epoch=batches_consumed_in_epoch,
        cfg=cfg,
    )
    _log_tracker_metrics(
        accelerator,
        {"train/completed": 1},
        step=global_step,
        enabled=tracking_enabled,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
