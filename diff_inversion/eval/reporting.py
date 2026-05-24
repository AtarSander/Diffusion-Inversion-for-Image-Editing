from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from loguru import logger
import torch


def summarize_numeric_rows(
    rows: list[dict[str, Any]],
    prefixes_to_skip: tuple[str, ...] = (),
) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith(prefixes_to_skip) or isinstance(value, bool):
                continue
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values_by_key.setdefault(key, []).append(float(value))

    summary = {}
    for key, values in values_by_key.items():
        values_t = torch.tensor(values, dtype=torch.float32)
        summary[f"{key}/mean"] = float(values_t.mean().item())
        summary[f"{key}/std"] = float(values_t.std(unbiased=False).item())
        summary[f"{key}/min"] = float(values_t.min().item())
        summary[f"{key}/max"] = float(values_t.max().item())
    return summary


def flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float | int | str | bool]:
    flat: dict[str, float | int | str | bool] = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, full_key))
        elif isinstance(value, (float, int, str, bool)):
            flat[full_key] = value
    return flat


def build_samples_table_rows(
    results: dict[str, Any],
) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for sample_name, sample in results["samples"].items():
        row: dict[str, float | int | str | bool] = {"sample": sample_name}
        row.update(flatten_metrics(sample["metadata"], prefix="metadata"))
        row.update(flatten_metrics(sample["trajectory"], prefix="trajectory"))
        row.update(flatten_metrics(sample["initial_latent_stats"], prefix="initial_latent"))
        row.update(flatten_metrics(sample["final_latent_stats"], prefix="final_latent"))
        if "first_pred_noise_stats" in sample:
            row.update(
                flatten_metrics(sample["first_pred_noise_stats"], prefix="first_pred_noise")
            )
        if "last_pred_noise_stats" in sample:
            row.update(flatten_metrics(sample["last_pred_noise_stats"], prefix="last_pred_noise"))
        if "inverted_noise_stats" in sample:
            row.update(flatten_metrics(sample["inverted_noise_stats"], prefix="inverted_noise"))
        if "inversion_error" in sample:
            row.update(flatten_metrics(sample["inversion_error"], prefix="inversion_error"))
        if "inversion_error_stats" in sample:
            row.update(
                flatten_metrics(sample["inversion_error_stats"], prefix="inversion_error_stats")
            )
        if "inversion_error_per_channel" in sample:
            row.update(
                flatten_metrics(
                    sample["inversion_error_per_channel"],
                    prefix="inversion_error_per_channel",
                )
            )
        if "inversion_error_structure" in sample:
            row.update(
                flatten_metrics(
                    sample["inversion_error_structure"],
                    prefix="inversion_error_structure",
                )
            )
        if "noise_normality" in sample:
            row.update(flatten_metrics(sample["noise_normality"], prefix="noise_normality"))
        if "reconstruction_image" in sample:
            row.update(
                flatten_metrics(
                    sample["reconstruction_image"],
                    prefix="reconstruction_image",
                )
            )
        rows.append(row)
    return rows


def sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return sanitized.strip("-.") or "evaluation-summary"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def log_to_wandb(results: dict[str, Any], output_dir: Path, config: Any) -> None:
    wandb_cfg = _config_get(config, "wandb", {})
    wandb_mode = _config_get(wandb_cfg, "mode", "disabled")
    if wandb_mode == "disabled":
        return

    try:
        import wandb
    except ImportError:
        logger.warning(
            "W&B logging skipped because `wandb` is not installed in the active environment"
        )
        return

    wandb_root = output_dir / "wandb"
    cache_dir = wandb_root / "cache"
    config_dir = wandb_root / "config"
    data_dir = wandb_root / "data"
    files_dir = wandb_root / "files"
    for directory in (wandb_root, cache_dir, config_dir, data_dir, files_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", wandb_root.as_posix())
    os.environ.setdefault("WANDB_CACHE_DIR", cache_dir.as_posix())
    os.environ.setdefault("WANDB_CONFIG_DIR", config_dir.as_posix())
    os.environ.setdefault("WANDB_DATA_DIR", data_dir.as_posix())

    try:
        wandb_config = {
            "input_dir": results["input_dir"],
            "output_dir": output_dir.as_posix(),
            "num_samples": results["num_samples"],
            "patch_size": _config_get(config, "patch_size"),
            "top_k": _config_get(config, "top_k"),
            "max_elements": _config_get(config, "max_elements"),
            "plain_threshold": _config_get(config, "plain_threshold"),
            "normality_sample_size": _config_get(config, "normality_sample_size"),
            "qq_num_quantiles": _config_get(config, "qq_num_quantiles"),
        }
        run = wandb.init(
            project=_config_get(wandb_cfg, "project"),
            entity=_config_get(wandb_cfg, "entity"),
            group=_config_get(wandb_cfg, "group"),
            name=_config_get(wandb_cfg, "run_name"),
            tags=_config_get(wandb_cfg, "tags"),
            job_type="evaluation",
            mode=wandb_mode,
            config=wandb_config,
            dir=wandb_root.as_posix(),
            settings=wandb.Settings(
                root_dir=wandb_root.as_posix(),
                x_files_dir=files_dir.as_posix(),
                x_disable_stats=True,
            ),
        )

        aggregate_metrics = flatten_metrics(results["aggregate"], prefix="aggregate")
        aggregate_metrics["num_samples"] = results["num_samples"]
        wandb.log(aggregate_metrics)

        rows = build_samples_table_rows(results)
        if rows:
            columns = sorted({key for row in rows for key in row})
            table = wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*[row.get(column) for column in columns])
            wandb.log({"samples_table": table})

        summary_json = output_dir / "evaluation_summary.json"
        summary_md = output_dir / "evaluation_summary.md"
        artifact = wandb.Artifact(
            name=sanitize_artifact_name(
                _config_get(wandb_cfg, "artifact_name") or output_dir.name
            ),
            type="evaluation",
        )
        if summary_json.exists():
            artifact.add_file(summary_json.as_posix())
        if summary_md.exists():
            artifact.add_file(summary_md.as_posix())
        for dirname in (
            "noise_comparisons",
            "normality",
            "image_comparisons",
            "inversion_diagnostics",
        ):
            artifact_dir = output_dir / dirname
            if artifact_dir.exists():
                artifact.add_dir(artifact_dir.as_posix(), name=dirname)
        run.log_artifact(artifact)

        run.summary["input_dir"] = results["input_dir"]
        run.summary["notes"] = results["notes"]
        run.finish()
        logger.success("Logged evaluation to Weights & Biases")
    except Exception as exc:
        logger.warning("W&B logging skipped because initialization failed: {}", exc)


def _write_rows(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    columns = sorted({key for row in rows for key in row})
    with (output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "evaluation_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    noise_comparisons = results.get("noise_comparisons") or []
    if noise_comparisons:
        _write_rows(output_dir / "noise_comparisons", "noise_comparisons", noise_comparisons)

    normality_comparisons = results.get("normality_comparisons") or []
    if normality_comparisons:
        _write_rows(output_dir / "normality", "normality_comparisons", normality_comparisons)

    image_comparisons = results.get("image_comparisons") or []
    if image_comparisons:
        _write_rows(output_dir / "image_comparisons", "image_comparisons", image_comparisons)

    inversion_diagnostics = results.get("inversion_diagnostics") or {}
    prediction_error = inversion_diagnostics.get("prediction_error") or {}
    prediction_by_step = prediction_error.get("by_step") or []
    if prediction_by_step:
        _write_rows(
            output_dir / "inversion_diagnostics",
            "prediction_error_by_step",
            prediction_by_step,
        )

    aggregate = results["aggregate"]
    markdown_path = output_dir / "evaluation_summary.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"- Input directory: `{results['input_dir']}`\n")
        f.write(f"- Samples: {results['num_samples']}\n\n")
        if noise_comparisons:
            f.write("- Noise comparison artifacts: `noise_comparisons/`\n\n")
        if normality_comparisons:
            f.write("- Normality artifacts: `normality/`\n\n")
        if image_comparisons:
            f.write("- Image reconstruction artifacts: `image_comparisons/`\n\n")
        if inversion_diagnostics.get("enabled"):
            f.write("- Inversion diagnostics: `inversion_diagnostics/`\n\n")
        f.write("## Aggregate\n\n")
        for name, metrics in aggregate.items():
            f.write(f"### {name}\n\n")
            for key, value in metrics.items():
                f.write(f"- `{key}`: {value}\n")
            f.write("\n")
        if inversion_diagnostics.get("enabled"):
            f.write("## Inversion Diagnostics\n\n")
            f.write(f"- Selected samples: {inversion_diagnostics.get('selected_samples')}\n")
            f.write(f"- Evaluated samples: {inversion_diagnostics.get('evaluated_samples')}\n")
            latent_location = inversion_diagnostics.get("latent_location") or {}
            if latent_location:
                f.write(
                    "- Latent-location heatmap: "
                    f"`{latent_location.get('plot_path') or latent_location.get('csv_path')}`\n"
                )
            if prediction_error:
                f.write(
                    "- Prediction error by step: "
                    f"`{prediction_error.get('plot_path') or prediction_error.get('summary_csv_path')}`\n"
                )
            warnings = inversion_diagnostics.get("warnings") or []
            if warnings:
                f.write(f"- Warnings: {len(warnings)} samples skipped\n")
            f.write("\n")
        f.write("## Notes\n\n")
        for note in results["notes"]:
            f.write(f"- {note}\n")

    logger.success("Saved {}", json_path)
    logger.success("Saved {}", markdown_path)
