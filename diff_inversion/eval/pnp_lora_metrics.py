"""Evaluate generated LoRA+PnP PIE-Bench images and summarize the metrics."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _as_str_list(values) -> list[str]:
    return [str(value) for value in values or []]


def _write_summary(result_path: Path, summary_path: Path) -> None:
    with result_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary: dict[str, object] = {"num_images": len(rows), "metrics": {}}
    for field in (rows[0].keys() if rows else []):
        if field == "file_id":
            continue
        values = []
        for row in rows:
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        summary["metrics"][field] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@hydra.main(config_path="../../config", config_name="eval/pnp_lora_metrics", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("PnP LoRA metrics config:\n{}", OmegaConf.to_yaml(cfg))

    repo_dir = _resolve_path(cfg.repo_dir)
    pnp_dir = repo_dir / "pnp_inversion"
    python = Path(to_absolute_path(str(cfg.python)))
    data_path = _resolve_path(cfg.data_path)
    generated_root = _resolve_path(cfg.generated_root)
    result_path = _resolve_path(cfg.result_path)
    summary_path = _resolve_path(cfg.summary_path)
    method_name = str(cfg.method_name)
    generated_images_dir = generated_root / method_name / "annotation_images"

    mapping_path = data_path / "mapping_file.json"
    source_images_dir = data_path / "annotation_images"
    if not pnp_dir.exists():
        raise FileNotFoundError(f"PnP directory does not exist: {pnp_dir}")
    if not python.is_file():
        raise FileNotFoundError(f"Metrics Python executable does not exist: {python}")
    if not mapping_path.exists() or not source_images_dir.exists():
        raise FileNotFoundError(f"Invalid PIE-Bench data path: {data_path}")
    if not generated_images_dir.exists():
        raise FileNotFoundError(f"Generated image directory does not exist: {generated_images_dir}")

    with mapping_path.open(encoding="utf-8") as f:
        mapping = json.load(f)
    categories = set(_as_str_list(cfg.edit_category_list))
    missing = [
        item["image_path"]
        for item in mapping.values()
        if str(item["editing_type_id"]) in categories
        and not (generated_images_dir / item["image_path"]).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} generated {method_name} images under {generated_images_dir}; "
            f"first: {preview}"
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        "-m",
        "evaluation.evaluate",
        "--annotation_mapping_file",
        str(mapping_path),
        "--src_image_folder",
        str(source_images_dir),
        "--tgt_image_folder",
        f"{method_name}={generated_images_dir}",
        "--result_path",
        str(result_path),
        "--device",
        str(cfg.device),
        "--metrics",
        *_as_str_list(cfg.metrics),
        "--edit_category_list",
        *_as_str_list(cfg.edit_category_list),
    ]

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/net/tscratch/people/plgatarsander/hf-cache")
    env.setdefault("MPLCONFIGDIR", "/net/tscratch/people/plgatarsander/matplotlib-cache")
    env.setdefault("TORCH_HOME", "/net/tscratch/people/plgatarsander/torch-cache")
    for directory_var in ("HF_HOME", "MPLCONFIGDIR", "TORCH_HOME"):
        Path(env[directory_var]).mkdir(parents=True, exist_ok=True)

    logger.info("Running PnP metrics from {}: {}", pnp_dir, " ".join(cmd))
    subprocess.run(cmd, cwd=pnp_dir, env=env, check=True)
    _write_summary(result_path, summary_path)
    logger.info("Wrote metrics CSV to {} and summary to {}", result_path, summary_path)


if __name__ == "__main__":
    main()
