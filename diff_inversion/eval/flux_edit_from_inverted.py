"""Generate FLUX edits directly from latents saved by a completed inversion run.

This is project code and reuses the shared FLUX sampling implementation. It does not
repeat inversion or reconstruction and does not implement an external editing method.
The target-prompt offset and CLIP protocol match ``flux_baseline.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import FluxPipeline
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.data.generate_flux_inversion_data import load_prompt_records
from diff_inversion.eval.flux_baseline import (
    _add_flux_clip_metrics,
    _execution_device,
    _link_source_file,
    _load_flux_pipeline,
    _load_tensor,
    _log_flux_results_to_wandb,
    _save_json,
    _validate_source_run_config,
    _validate_source_sample,
    build_flux_edit_jobs,
    build_flux_prompt_jobs,
)
from diff_inversion.eval.reporting import flatten_metrics, summarize_numeric_rows
from diff_inversion.eval.sample_metrics import image_pair_metrics
from diff_inversion.modeling.flux_sampling import (
    decode_flux_latents,
    encode_flux_prompt,
    prepare_flux_latents,
    sample_flux_latent,
)


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _edit_sample_is_complete(sample_dir: Path) -> bool:
    return all(
        (sample_dir / name).exists()
        for name in ("original.png", "edited.png", "prompt.json", "metrics.json")
    )


@torch.no_grad()
def _generate_edit(
    pipe: FluxPipeline,
    *,
    source_sample_dir: Path,
    output_sample_dir: Path,
    sample_index: int,
    prompt: str,
    target_prompt: str,
    seed: int,
    cfg: DictConfig,
) -> dict[str, Any]:
    _validate_source_sample(
        source_sample_dir,
        expected_prompt=prompt,
        expected_seed=seed,
    )
    inverted_noise_path = source_sample_dir / "inverted_noise.pt"
    if not inverted_noise_path.exists():
        raise FileNotFoundError(f"Missing recovered noise: {inverted_noise_path}")

    device = _execution_device(pipe)
    generated_noise, latent_image_ids = prepare_flux_latents(
        pipe,
        height=int(cfg.height),
        width=int(cfg.width),
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    inverted_noise = _load_tensor(inverted_noise_path)
    if inverted_noise.shape != generated_noise.shape:
        raise ValueError(
            f"Recovered-noise shape mismatch in {source_sample_dir}: "
            f"stored={tuple(inverted_noise.shape)}, expected={tuple(generated_noise.shape)}"
        )
    inverted_noise = inverted_noise.to(device=device)

    target_conditioning = encode_flux_prompt(
        pipe,
        prompt=target_prompt,
        max_sequence_length=int(cfg.max_sequence_length),
        device=device,
    )
    edited_latent, _, _, _ = sample_flux_latent(
        pipe,
        initial_latents=inverted_noise,
        conditioning=target_conditioning,
        latent_image_ids=latent_image_ids,
        num_inference_steps=int(cfg.num_inference_steps),
        guidance_scale=float(cfg.editing.guidance_scale),
        save_trajectory=False,
        save_velocities=False,
        progress_desc=f"Editing {output_sample_dir.name}",
    )
    edited_image = decode_flux_latents(
        pipe,
        edited_latent,
        height=int(cfg.height),
        width=int(cfg.width),
    )

    output_sample_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_sample_dir / "original.png"
    edited_path = output_sample_dir / "edited.png"
    _link_source_file(source_sample_dir / "original.png", original_path)
    _link_source_file(inverted_noise_path, output_sample_dir / "inverted_noise.pt")
    edited_image.save(edited_path)

    metrics = {
        "editing": {
            "settings": {
                "source_prompt": prompt,
                "target_prompt": target_prompt,
                "guidance_scale": float(cfg.editing.guidance_scale),
                "source_inversion_dir": source_sample_dir.parent.as_posix(),
            },
            "inverted_noise": {
                "image": image_pair_metrics(
                    original_path,
                    edited_path,
                    plain_threshold=float(cfg.metrics.plain_threshold),
                )
            },
        }
    }
    _save_json(
        output_sample_dir / "prompt.json",
        {
            "sample_index": sample_index,
            "prompt": prompt,
            "target_prompt": target_prompt,
            "seed": seed,
        },
    )
    _save_json(output_sample_dir / "metrics.json", metrics)
    logger.success("Generated edit for {}", output_sample_dir.name)
    return metrics


@hydra.main(
    config_path="../../config",
    config_name="eval/flux_edit_from_inverted",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if not bool(cfg.editing.enabled):
        raise ValueError("flux_edit_from_inverted requires editing.enabled=true")
    if cfg.editing.guidance_scale is None:
        raise ValueError("editing.guidance_scale must be set explicitly")

    source_dir = _resolve_path(cfg.source_dir)
    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_source_run_config(source_dir, cfg)

    config_path = output_dir / "run_config.yaml"
    if config_path.exists():
        existing = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        current = OmegaConf.to_container(cfg, resolve=True)
        if existing != current:
            raise ValueError(f"Refusing to mix configurations in {output_dir}")
    OmegaConf.save(cfg, config_path)

    prompts = load_prompt_records(
        cfg.prompts_jsonl,
        list(cfg.prompts),
        prompt_key=str(cfg.prompt_key),
    )
    prompt_jobs = build_flux_prompt_jobs(
        prompts,
        samples_per_prompt=int(cfg.samples_per_prompt),
        start_index=int(cfg.start_index),
        num_samples=None if cfg.num_samples is None else int(cfg.num_samples),
    )
    edit_jobs = build_flux_edit_jobs(
        prompt_jobs,
        target_prompts=prompts,
        target_prompt_template=None,
        target_prompt_offset=int(cfg.editing.target_prompt_offset),
    )

    device_name = str(cfg.device).strip().lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if bool(cfg.model.require_cuda) and device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX edit generation")
    pipe = _load_flux_pipeline(cfg, device)

    sample_results: dict[str, Any] = {}
    for sample_index, prompt, target_prompt in edit_jobs:
        if not target_prompt:
            raise ValueError(f"Missing target prompt for sample {sample_index}")
        sample_name = f"sample_{sample_index:06d}"
        sample_dir = output_dir / sample_name
        if _edit_sample_is_complete(sample_dir) and not bool(cfg.overwrite):
            with (sample_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                sample_results[sample_name] = json.load(handle)
            logger.info("Skipping completed edit: {}", sample_dir)
            continue
        sample_results[sample_name] = _generate_edit(
            pipe,
            source_sample_dir=source_dir / sample_name,
            output_sample_dir=sample_dir,
            sample_index=sample_index,
            prompt=prompt,
            target_prompt=str(target_prompt),
            seed=int(cfg.seed) + sample_index,
            cfg=cfg,
        )

    clip_rows = _add_flux_clip_metrics(
        output_dir=output_dir,
        sample_results=sample_results,
        clip_config=cfg.metrics.clip,
    )
    aggregate = summarize_numeric_rows(
        [flatten_metrics(metrics) for metrics in sample_results.values()]
    )
    results = {
        "model_id": str(cfg.model.model_id),
        "guidance_scale": float(cfg.guidance_scale),
        "edit_guidance_scale": float(cfg.editing.guidance_scale),
        "source_dir": source_dir.as_posix(),
        "start_index": int(cfg.start_index),
        "requested_samples": len(edit_jobs),
        "num_samples": len(sample_results),
        "samples": sample_results,
        "clip_text_alignments": clip_rows,
        "aggregate": aggregate,
    }
    _save_json(output_dir / "results.json", results)
    _log_flux_results_to_wandb(results, output_dir=output_dir, cfg=cfg)
    logger.success("Saved {} FLUX edits to {}", len(sample_results), output_dir)


if __name__ == "__main__":
    main()
