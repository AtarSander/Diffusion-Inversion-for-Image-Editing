"""Small readable experiment configuration; paths are relative to the YAML file."""

import json
from pathlib import Path

from omegaconf import OmegaConf


def load_config(path):
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    source = Path(config["source_manifest"])
    config["source_manifest"] = str((path.parent / source).resolve())
    model_id = config["model"]["id"]
    if model_id.startswith((".", "/")):
        config["model"]["id"] = str((path.parent / model_id).resolve())
    if config.get("lora"):
        checkpoint = Path(config["lora"]["checkpoint_path"])
        config["lora"]["checkpoint_path"] = str((path.parent / checkpoint).resolve())
    if config["reader"] not in ("free", "direct"):
        raise ValueError("reader must be free or direct")
    if config["steps"] < 2 or config["size"] < 8 or config["size"] % 8:
        raise ValueError("Use >=2 denoising steps and an image size divisible by 8")
    if not config["generation"]["prompts"] or not config["generation"]["seeds"]:
        raise ValueError("Freeze a nonempty common prompt set and seed list")
    return config


def read_sources(path):
    """Only construction examples belong in this explicitly selected source list."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
