"""Evaluate generated LoRA+PnP PIE-Bench images and summarize the metrics."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

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
    python_config = cfg.get("python")
    # Keep the virtual-environment executable itself. Resolving its symlink points
    # at the base interpreter and drops the environment's site-packages.
    python = Path(to_absolute_path(str(python_config))) if python_config else Path(sys.executable)
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

    cache_root = _resolve_path(cfg.get("cache_root", repo_dir / ".cache" / "metrics"))
    cache_directories = {
        "HF_HOME": _resolve_path(cfg.get("hf_home", cache_root / "huggingface")),
        "MPLCONFIGDIR": _resolve_path(cfg.get("matplotlib_cache", cache_root / "matplotlib")),
        "TORCH_HOME": _resolve_path(cfg.get("torch_home", cache_root / "torch")),
    }
    env = os.environ.copy()
    for directory_var, directory in cache_directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        env[directory_var] = str(directory)

    logger.info("Running PnP metrics from {}: {}", pnp_dir, " ".join(cmd))
    subprocess.run(cmd, cwd=pnp_dir, env=env, check=True)
    _write_summary(result_path, summary_path)
    logger.info("Wrote metrics CSV to {} and summary to {}", result_path, summary_path)


if __name__ == "__main__":
    main()
