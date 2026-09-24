"""Evaluate named matched TADA artifacts through the standard SDXL evaluator."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diff_inversion.eval.reporting import write_outputs
from diff_inversion.eval.run import run_evaluation


def _link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.symlink_to(source.resolve())


def build_evaluation_view(
    source_root: Path,
    view_root: Path,
    noise_name: str,
    image_name: str,
    expected_samples: int,
) -> int:
    """Create a node-local, inode-cheap view using the evaluator's standard names."""
    sample_dirs = sorted(path for path in source_root.glob("sample_*") if path.is_dir())
    if len(sample_dirs) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples in {source_root}, found {len(sample_dirs)}"
        )
    if view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True)

    for sample_dir in sample_dirs:
        view_sample = view_root / sample_dir.name
        view_latents = view_sample / "latents"
        view_latents.mkdir(parents=True)
        _link(sample_dir / "latents" / "initial_noise.pt", view_latents / "x_000.pt")
        _link(sample_dir / "latents" / "final_latent.pt", view_latents / "x_001.pt")
        _link(sample_dir / "final.png", view_sample / "final.png")
        _link(sample_dir / noise_name, view_sample / "inverted_noise.pt")
        _link(sample_dir / image_name, view_sample / "reconstructed.png")
        for metadata_name in ("prompt.json", "meta.json", "timesteps.json"):
            metadata_path = sample_dir / metadata_name
            if metadata_path.exists():
                _link(metadata_path, view_sample / metadata_name)
    return len(sample_dirs)


@hydra.main(
    config_path="../../config",
    config_name="eval/tada_matched_metrics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info("Matched TADA metric config:\n{}", OmegaConf.to_yaml(cfg))
    input_dir = Path(to_absolute_path(str(cfg.input_dir))).resolve()
    output_root = Path(to_absolute_path(str(cfg.output_root))).resolve()
    runtime_root = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")))
    runtime_root = runtime_root / "sdxl_tada_matched_metric_views"
    variants = OmegaConf.to_container(cfg.variants, resolve=True)
    if not isinstance(variants, list):
        raise TypeError("variants must be a list")

    for variant in variants:
        name = str(variant["name"])
        view_root = runtime_root / name
        count = build_evaluation_view(
            source_root=input_dir,
            view_root=view_root,
            noise_name=str(variant["noise_name"]),
            image_name=str(variant["image_name"]),
            expected_samples=int(cfg.expected_samples),
        )
        output_dir = output_root / name
        logger.info("Evaluating {} matched samples for {}", count, name)
        results = run_evaluation(
            input_dir=view_root,
            output_dir=output_dir,
            max_samples=None,
            patch_size=int(cfg.patch_size),
            top_k=int(cfg.top_k),
            max_elements=int(cfg.max_elements),
            save_noise_previews=False,
            max_preview_samples=0,
            plain_threshold=float(cfg.plain_threshold),
            normality_sample_size=int(cfg.normality_sample_size),
            qq_num_quantiles=int(cfg.qq_num_quantiles),
            save_normality_plots=False,
            calculate_lpips=bool(cfg.calculate_lpips),
            lpips_device=str(cfg.lpips_device),
            lpips_batch_size=int(cfg.lpips_batch_size),
            reconstruction_image_name="reconstructed.png",
            clip_text_alignment={"enabled": False},
            inversion_diagnostics={"enabled": False},
        )
        results["matched_variant"] = name
        results["source_input_dir"] = input_dir.as_posix()
        results["source_noise_name"] = str(variant["noise_name"])
        results["source_reconstruction_name"] = str(variant["image_name"])
        write_outputs(results, output_dir)
        shutil.rmtree(view_root)

    logger.success("Finished all matched TADA metrics in {}", output_root)


if __name__ == "__main__":
    main()
