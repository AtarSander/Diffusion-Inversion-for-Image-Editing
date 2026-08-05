"""Run P2P editing evaluation with LoRA-assisted SD1.5 inversion."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.eval.runtime import build_subprocess_environment


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _as_str_list(values) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values]


@hydra.main(config_path="../../config", config_name="eval/p2p_lora", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("P2P LoRA eval config:\n{}", OmegaConf.to_yaml(cfg))

    repo_dir = _resolve_path(cfg.repo_dir)
    pnp_dir = repo_dir / "pnp_inversion"
    data_path = _resolve_path(cfg.data_path)
    output_path = _resolve_path(cfg.output_path)
    lora_checkpoint = _resolve_path(cfg.lora.checkpoint_path)
    cache_root_config = OmegaConf.select(cfg, "cache_root", default=None)
    cache_root = (
        _resolve_path(cache_root_config)
        if cache_root_config
        else repo_dir / ".cache" / "editing-eval"
    )

    if not pnp_dir.exists():
        raise FileNotFoundError(f"P2P directory does not exist: {pnp_dir}")
    if not (data_path / "mapping_file.json").exists():
        raise FileNotFoundError(f"P2P mapping file does not exist: {data_path / 'mapping_file.json'}")
    if not lora_checkpoint.exists():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {lora_checkpoint}")

    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_editing_p2p.py",
        "--data_path",
        str(data_path),
        "--output_path",
        str(output_path),
        "--edit_category_list",
        *_as_str_list(cfg.edit_category_list),
        "--edit_method_list",
        *_as_str_list(cfg.edit_method_list),
        "--lora_checkpoint",
        str(lora_checkpoint),
        "--lora_rank",
        str(cfg.lora.r),
        "--lora_alpha",
        str(cfg.lora.lora_alpha),
        "--lora_dropout",
        str(cfg.lora.lora_dropout),
        "--lora_scale",
        str(cfg.lora.scale),
        "--inversion_guidance_scale",
        str(OmegaConf.select(cfg, "guidance.inversion_scale", default=1.0)),
    ]
    if bool(cfg.rerun_exist_images):
        cmd.append("--rerun_exist_images")

    env = build_subprocess_environment(cache_root)

    logger.info("Running P2P command from {}: {}", pnp_dir, " ".join(cmd))
    subprocess.run(cmd, cwd=pnp_dir, env=env, check=True)


if __name__ == "__main__":
    main()
