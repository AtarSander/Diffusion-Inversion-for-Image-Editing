"""Run SD1.5 DDIM/Direct-Inversion controls for the PIE-Bench editors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.eval.runtime import build_subprocess_environment

_BACKENDS = {
    "p2p": ("run_editing_p2p.py", "p2p"),
    "pnp": ("run_editing_pnp.py", "pnp"),
    "masactrl": ("run_editing_masactrl.py", "masactrl"),
    "pix2pix_zero": ("run_editing_pix2pix_zero.py", "pix2pix-zero"),
}
_INVERSIONS = {"ddim", "directinversion"}


def _resolve_path(path: str | Path) -> Path:
    return Path(to_absolute_path(str(path))).resolve()


def _as_str_list(values) -> list[str]:
    return [str(value) for value in values]


@hydra.main(config_path="../../config", config_name="eval/sd15_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("SD1.5 baseline eval config:\n{}", OmegaConf.to_yaml(cfg))

    backend = str(cfg.backend)
    inversion = str(cfg.inversion)
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose one of {sorted(_BACKENDS)}")
    if inversion not in _INVERSIONS:
        raise ValueError(
            f"Unknown inversion {inversion!r}; choose one of {sorted(_INVERSIONS)}"
        )
    if str(cfg.model_id) != "runwayml/stable-diffusion-v1-5":
        raise ValueError("This control runner is intentionally restricted to SD1.5")
    if str(cfg.inference_dtype).lower() not in {"fp16", "float16"}:
        raise ValueError("All four controls must use FP16 to match existing LoRA inference")

    repo_dir = _resolve_path(cfg.repo_dir)
    pnp_dir = repo_dir / "pnp_inversion"
    data_path = _resolve_path(cfg.data_path)
    output_path = _resolve_path(cfg.output_path)
    cache_root = _resolve_path(cfg.cache_root)
    if not (data_path / "mapping_file.json").exists():
        raise FileNotFoundError(f"PIE-Bench mapping file is missing under {data_path}")

    script, method_suffix = _BACKENDS[backend]
    method = f"{inversion}+{method_suffix}"
    cmd = [
        sys.executable,
        script,
        "--data_path",
        str(data_path),
        "--output_path",
        str(output_path),
        "--edit_category_list",
        *_as_str_list(cfg.edit_category_list),
        "--edit_method_list",
        method,
        "--inversion_guidance_scale",
        str(cfg.inversion_guidance_scale),
    ]

    # PnP is already fixed to SD1.5. The other backends expose the model explicitly.
    if backend != "pnp":
        cmd.extend(["--model_key", str(cfg.model_id)])
    if bool(cfg.rerun_exist_images):
        cmd.append("--rerun_exist_images")

    output_path.mkdir(parents=True, exist_ok=True)
    env = build_subprocess_environment(cache_root)
    # P2P supports multiple dtypes; pin it so an inherited shell variable cannot
    # make this control differ from the completed LoRA evaluations.
    env["P2P_DTYPE"] = "fp16"

    logger.info("Running {} from {}: {}", method, pnp_dir, " ".join(cmd))
    subprocess.run(cmd, cwd=pnp_dir, env=env, check=True)


if __name__ == "__main__":
    main()
