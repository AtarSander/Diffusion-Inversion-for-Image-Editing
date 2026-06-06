"""Assign target prompts to generated samples for synthetic editing evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _sample_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("sample_*") if path.is_dir())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data)!r}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _source_prompts_from_samples(samples: list[Path]) -> list[str]:
    prompts = []
    for sample_dir in samples:
        prompt_path = sample_dir / "prompt.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing prompt metadata: {prompt_path}")
        record = _read_json(prompt_path)
        prompt = str(record.get("prompt") or "")
        if not prompt:
            raise ValueError(f"Missing `prompt` in {prompt_path}")
        prompts.append(prompt)
    return prompts


def _prompts_from_jsonl(path: Path, prompt_column: str) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if prompt_column not in record:
                raise KeyError(
                    f"Column {prompt_column!r} not found in {path}:{line_number}"
                )
            prompt = str(record[prompt_column]).strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def _pick_target_prompt(
    source_prompt: str,
    prompts: list[str],
    index: int,
    offset: int,
) -> tuple[str, int]:
    if not prompts:
        raise ValueError("No target prompts available")

    prompt_count = len(prompts)
    for advance in range(prompt_count):
        target_index = (index + offset + advance) % prompt_count
        target_prompt = prompts[target_index]
        if target_prompt != source_prompt:
            return target_prompt, target_index
    raise ValueError("Could not find a target prompt different from the source prompt")


@hydra.main(config_path="../../config", config_name="eval/assign_target_prompts", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Assign target prompts config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = _resolve_path(cfg.input_dir)
    samples = _sample_dirs(input_dir)
    if not samples:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    source_prompts = _source_prompts_from_samples(samples)
    prompts_jsonl = OmegaConf.select(cfg, "target_prompts_jsonl", default=None)
    if prompts_jsonl:
        target_prompts = _prompts_from_jsonl(
            _resolve_path(str(prompts_jsonl)),
            str(cfg.prompt_column),
        )
    else:
        target_prompts = source_prompts

    assigned = 0
    skipped = 0
    offset = int(cfg.target_offset)
    overwrite = bool(cfg.overwrite)
    for index, (sample_dir, source_prompt) in enumerate(zip(samples, source_prompts, strict=True)):
        prompt_path = sample_dir / "prompt.json"
        record = _read_json(prompt_path)
        if record.get("target_prompt") and not overwrite:
            skipped += 1
            continue

        target_prompt, target_index = _pick_target_prompt(
            source_prompt=source_prompt,
            prompts=target_prompts,
            index=index,
            offset=offset,
        )
        record["target_prompt"] = target_prompt
        record["target_prompt_index"] = target_index
        record["target_prompt_source"] = (
            Path(str(prompts_jsonl)).as_posix() if prompts_jsonl else "input_dir_shift"
        )
        _write_json(prompt_path, record)
        assigned += 1

    logger.success(
        "Assigned target prompts in {} samples from {} (skipped {})",
        assigned,
        input_dir,
        skipped,
    )


if __name__ == "__main__":
    main()

