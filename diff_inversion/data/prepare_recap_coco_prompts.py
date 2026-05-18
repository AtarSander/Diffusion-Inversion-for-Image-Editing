"""Prepare Recap-COCO prompt splits from the downloaded parquet dataset."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def normalize_prompt(text: str) -> str:
    """Collapse repeated whitespace and trim the prompt text."""
    return " ".join(text.strip().split())


def resolve_input_parquet_path(cfg: DictConfig) -> Path:
    """Resolve the Recap-COCO parquet path from Hydra config."""
    if cfg.prepare_input_parquet is not None:
        return Path(to_absolute_path(str(cfg.prepare_input_parquet)))

    return Path(to_absolute_path(str(cfg.parquet_path)))


def resolve_output_dir(cfg: DictConfig) -> Path:
    """Resolve the processed Recap-COCO prompt output directory from Hydra config."""
    if cfg.prepare_output_dir is not None:
        return Path(to_absolute_path(str(cfg.prepare_output_dir)))

    return Path(to_absolute_path(str(cfg.processed_dir)))


def config_list(value: Any) -> List[str]:
    """Convert a Hydra list-like value to a plain string list."""
    if value is None:
        return []

    resolved = OmegaConf.to_container(value, resolve=True)
    if resolved is None:
        return []

    return [str(item) for item in resolved]


def load_recap_coco_records(cfg: DictConfig, parquet_path: Path) -> List[Dict[str, Any]]:
    """Load Recap-COCO parquet rows and normalize them into prompt records."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Preparing Recap-COCO requires the `datasets` package. "
            "Run `uv sync` to install project dependencies."
        ) from exc

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split=str(cfg.split),
    )

    drop_columns = [
        column for column in config_list(cfg.drop_columns) if column in dataset.column_names
    ]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)

    keep_columns: Set[str] = set(config_list(cfg.keep_columns))
    prompt_column = str(cfg.prompt_column)
    if prompt_column not in dataset.column_names:
        raise KeyError(
            f"Prompt column '{prompt_column}' not found. Available columns: {dataset.column_names}"
        )

    records: List[Dict[str, Any]] = []
    for row in dataset:
        record = {
            key: row[key] for key in dataset.column_names if key in keep_columns and key in row
        }
        record["prompt"] = normalize_prompt(str(row[prompt_column]))
        records.append(record)

    return records


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate prompts while keeping the first occurrence of each one."""
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []

    for record in records:
        prompt = record["prompt"]
        if prompt in seen:
            continue
        seen.add(prompt)
        deduped.append(record)

    return deduped


def split_records(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split prompt records into train, validation, and test partitions."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    return train_records, val_records, test_records


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@hydra.main(config_path="../../config/data", config_name="recap_coco", version_base=None)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for transforming Recap-COCO parquet rows into prompt splits."""
    parquet_path = resolve_input_parquet_path(cfg)
    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {parquet_path}\n"
            "Run the download step first or override prepare_input_parquet."
        )

    records = load_recap_coco_records(cfg, parquet_path)
    logger.info("Loaded {} Recap-COCO records from {}", len(records), parquet_path.name)

    if cfg.deduplicate:
        before = len(records)
        records = deduplicate_records(records)
        after = len(records)
        logger.info("Deduplicated prompts: {} -> {}", before, after)

    rng = random.Random(cfg.seed)
    rng.shuffle(records)

    train_records, val_records, test_records = split_records(
        records=records,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )

    save_jsonl(records, output_dir / "all_prompts.jsonl")
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(val_records, output_dir / "val.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")

    summary = {
        "dataset": cfg.name,
        "input_parquet_path": str(parquet_path),
        "prompt_column": cfg.prompt_column,
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "deduplicate": cfg.deduplicate,
        "seed": cfg.seed,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.success("Saved prompts: {}", output_dir / "all_prompts.jsonl")
    logger.success("Saved prompts: {}", output_dir / "train.jsonl")
    logger.success("Saved prompts: {}", output_dir / "val.jsonl")
    logger.success("Saved prompts: {}", output_dir / "test.jsonl")
    logger.success("Saved summary: {}", output_dir / "summary.json")


if __name__ == "__main__":
    main()
