"""Runtime environment helpers shared by editing evaluation entrypoints."""

from __future__ import annotations

import os
from pathlib import Path


def build_subprocess_environment(cache_root: Path) -> dict[str, str]:
    """Build a deterministic subprocess environment rooted in Hydra storage."""
    env = os.environ.copy()
    cache_directories = {
        "HF_HOME": "huggingface",
        "TORCH_HOME": "torch",
        "MPLCONFIGDIR": "matplotlib",
        "WANDB_CACHE_DIR": "wandb-cache",
        "WANDB_DIR": "wandb",
    }
    for variable, relative_path in cache_directories.items():
        directory = cache_root / relative_path
        directory.mkdir(parents=True, exist_ok=True)
        env[variable] = str(directory)
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env
