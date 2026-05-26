from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch

from diff_inversion.eval.clip_dino import (
    get_clip,
    get_clip_features,
    output_to_feature_tensor,
    resolve_device,
)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    return getattr(config, key, default)


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features.float(), dim=1, eps=1e-12)


def _clip_score(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(image_features * text_features, dim=1)
    return 250.0 * cosine.clamp_min(0.0)


def _directional_clip_score(
    source_image_features: torch.Tensor,
    candidate_image_features: torch.Tensor,
    source_text_features: torch.Tensor,
    target_text_features: torch.Tensor,
) -> torch.Tensor:
    image_direction = _normalize(candidate_image_features - source_image_features)
    text_direction = _normalize(target_text_features - source_text_features)
    return 100.0 * torch.sum(image_direction * text_direction, dim=1)


def _open_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _text_features(
    prompts: list[str],
    clip_processor,
    clip_model,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = clip_processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_features = output_to_feature_tensor(clip_model.get_text_features(**inputs))
            features.append(text_features.detach().cpu())
    return torch.cat(features)


def add_clip_text_alignment_metrics(
    image_comparisons: list[dict[str, Any]],
    per_sample: dict[str, Any],
    config: Any,
) -> list[dict[str, Any]]:
    if not bool(_config_get(config, "enabled", False)):
        return []
    if not image_comparisons:
        return []

    target_prompt = _config_get(config, "target_prompt", None)
    if target_prompt is not None:
        target_prompt = str(target_prompt)
    model_name = str(_config_get(config, "model_name", "openai/clip-vit-base-patch32"))
    device = resolve_device(str(_config_get(config, "device", "auto")))
    batch_size = int(_config_get(config, "batch_size", 64))

    rows = []
    for row in image_comparisons:
        source_prompt = str(row.get("source_prompt") or row.get("prompt") or "")
        row_target_prompt = row.get("target_prompt") or target_prompt
        row_target_prompt = str(row_target_prompt) if row_target_prompt else ""
        if not source_prompt:
            continue
        rows.append(
            {
                "sample": row["sample"],
                "source_prompt": source_prompt,
                "target_prompt": row_target_prompt,
                "source_image_path": row["final_image_path"],
                "candidate_image_path": row["reconstructed_image_path"],
            }
        )

    if not rows:
        logger.warning("CLIP text alignment skipped: no source prompts found")
        return []

    logger.info(
        "Calculating CLIP text alignment with {} on {} for {} samples",
        model_name,
        device,
        len(rows),
    )
    processor, model = get_clip(model_name, device=device)
    model.eval()

    source_images = _open_images([Path(str(row["source_image_path"])) for row in rows])
    candidate_images = _open_images([Path(str(row["candidate_image_path"])) for row in rows])
    source_prompts = [str(row["source_prompt"]) for row in rows]

    source_image_features = _normalize(
        get_clip_features(source_images, processor, model, batch_size=batch_size, device=device)
    )
    candidate_image_features = _normalize(
        get_clip_features(candidate_images, processor, model, batch_size=batch_size, device=device)
    )
    source_text_features = _normalize(
        _text_features(source_prompts, processor, model, batch_size=batch_size, device=device)
    )

    source_scores = _clip_score(candidate_image_features, source_text_features)
    for row, source_score in zip(rows, source_scores, strict=True):
        row["clip_source"] = float(source_score.item())

    target_rows = [idx for idx, row in enumerate(rows) if row["target_prompt"]]
    if target_rows:
        target_prompts = [str(rows[idx]["target_prompt"]) for idx in target_rows]
        target_text_features = _normalize(
            _text_features(target_prompts, processor, model, batch_size=batch_size, device=device)
        )
        target_scores = _clip_score(
            candidate_image_features[target_rows],
            target_text_features,
        )
        directional_scores = _directional_clip_score(
            source_image_features[target_rows],
            candidate_image_features[target_rows],
            source_text_features[target_rows],
            target_text_features,
        )
        for idx, target_score, directional_score in zip(
            target_rows,
            target_scores,
            directional_scores,
            strict=True,
        ):
            rows[idx]["clip_target"] = float(target_score.item())
            rows[idx]["clip_directional"] = float(directional_score.item())
    else:
        logger.warning(
            "CLIP target and directional scores skipped: no target_prompt configured"
        )

    rows_by_sample = {str(row["sample"]): row for row in rows}
    for sample_name, row in rows_by_sample.items():
        per_sample.setdefault(sample_name, {})["clip_text_alignment"] = {
            key: value
            for key, value in row.items()
            if key
            in {
                "source_prompt",
                "target_prompt",
                "clip_source",
                "clip_target",
                "clip_directional",
            }
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows
