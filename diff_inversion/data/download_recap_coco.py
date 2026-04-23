"""Download the Recap-COCO dataset from Hugging Face using Hydra config."""

from pathlib import Path
from typing import List, Optional, Union

import hydra
from huggingface_hub import snapshot_download
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _optional_list(value: object) -> Optional[Union[List[str], str]]:
    """Convert Hydra list-like config values to plain containers for HF Hub."""
    if value is None:
        return None

    resolved = OmegaConf.to_container(value, resolve=True)
    if resolved is None or isinstance(resolved, str):
        return resolved

    if isinstance(resolved, list):
        return [str(item) for item in resolved]

    raise TypeError(f"Expected string, list, or null pattern value, got {type(resolved)}")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Download Recap-COCO files into a project-local directory."""
    output_dir = Path(to_absolute_path(str(cfg.data.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Hugging Face dataset snapshot")
    logger.info("repo_id={}", cfg.data.repo_id)
    logger.info("revision={}", cfg.data.revision)
    logger.info("output_dir={}", output_dir)

    saved_path = snapshot_download(
        repo_id=str(cfg.data.repo_id),
        repo_type=str(cfg.data.repo_type),
        revision=str(cfg.data.revision),
        local_dir=output_dir,
        allow_patterns=_optional_list(cfg.data.allow_patterns),
        ignore_patterns=_optional_list(cfg.data.ignore_patterns),
        token=cfg.data.token,
        force_download=bool(cfg.data.force_download),
        local_files_only=bool(cfg.data.local_files_only),
        max_workers=int(cfg.data.max_workers),
    )

    logger.success("Recap-COCO saved to {}", saved_path)


if __name__ == "__main__":
    main()
