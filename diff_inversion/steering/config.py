"""Load experiment YAML and source-image lists, resolving paths relative to each file."""

import json
import math
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from .types import Config, EditSource, Source


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    config = cast(Config, OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    source = Path(config["source_manifest"])
    config["source_manifest"] = str((path.parent / source).resolve())
    model_id = config["model"]["id"]
    if model_id.startswith((".", "/")):
        config["model"]["id"] = str((path.parent / model_id).resolve())
    if config.get("lora"):
        checkpoint = Path(config["lora"]["checkpoint_path"])
        config["lora"]["checkpoint_path"] = str((path.parent / checkpoint).resolve())
    if "reader" in config:
        raise ValueError("Remove 'reader': features are always read at saved inversion states")
    if config["steps"] < 2 or config["size"] < 8 or config["size"] % 8:
        raise ValueError("Use >=2 denoising steps and an image size divisible by 8")
    if "generation" in config:
        validate_generation(config)
    if "edit" in config:
        validate_edit(config)
    return config


def validate_steering(settings: dict, steps: int, section: str) -> None:
    """Reject ambiguous steering settings shared by the generation and edit blocks."""
    for key in ("representations", "controls", "strengths"):
        values = settings[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{section}.{key} must be nonempty and contain no duplicates")
    if set(settings["representations"]) - {"CA", "SA", "block"}:
        raise ValueError("Representations must be CA, SA or block")
    if set(settings["controls"]) - {"true", "swap", "gaussian"}:
        raise ValueError('Controls must be "true", "swap" or "gaussian"; quote "true" in YAML')
    for strength in settings["strengths"]:
        if type(strength) not in (int, float) or not math.isfinite(strength):
            raise ValueError(f"{section} strengths must be finite numbers")
    if type(settings.get("renorm", False)) is not bool:
        raise ValueError(f"{section}.renorm must be a boolean when present")
    start, stop = settings["start_call"], settings["stop_call"]
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= steps:
        raise ValueError("Use 0-based denoising calls in [0, steps), with exclusive stop_call")


def validate_generation(config: Config) -> None:
    """Reject ambiguous comparisons before loading the model or creating outputs."""
    settings = config["generation"]
    for key in ("prompts", "seeds"):
        values = settings[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"generation.{key} must be nonempty and contain no duplicates")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in settings["prompts"]):
        raise ValueError("Generation prompts must be nonempty strings")
    if any(type(seed) is not int for seed in settings["seeds"]):
        raise ValueError("Generation seeds must be integers")
    guidance = settings["guidance_scale"]
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance_scale must be a finite, nonnegative number")
    validate_steering(settings, config["steps"], "generation")


def validate_edit(config: Config) -> None:
    """Reject ambiguous edit settings: steering knobs applied to inverted images."""
    validate_steering(config["edit"], config["steps"], "edit")


def read_sources(path: str | Path) -> list[Source]:
    """Read construction images; pair kth positive with kth negative for the sign null."""
    path = Path(path)
    rows: list[Source] = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Source IDs must be unique")
    for row in rows:
        if row["label"] not in ("positive", "negative"):
            raise ValueError("Source labels must be positive/negative")
        if row.get("split", "construction") != "construction":
            raise ValueError("Held-out images cannot be used to fit directions")
        row["path"] = str((path.parent / row["path"]).resolve())
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    count = sum(row["label"] == "positive" for row in rows)
    if count < 2 or count % 2 or count * 2 != len(rows):
        raise ValueError("Use equal, even class sizes >=2; pair classes by their row order")
    return rows


def read_edit_sources(path: str | Path) -> list[EditSource]:
    """Read edit input images: jsonl rows {id, path}, paths relative to the manifest."""
    path = Path(path)
    rows: list[EditSource] = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Edit sources must be nonempty with unique IDs")
    for row in rows:
        row["path"] = str((path.parent / row["path"]).resolve())
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return rows


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
