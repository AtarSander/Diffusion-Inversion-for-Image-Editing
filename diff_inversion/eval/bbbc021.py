"""Run OVG's BBBC021 benchmark with an inversion-only LoRA adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from diff_inversion.eval.ovg_bridge import OVG_REVISION, load_upstream, make_inversion_method


def runner_class():
    import pandas as pd
    from PIL import Image
    import torch
    from diffusers import DDIMScheduler
    from peft import LoraConfig, set_peft_model_state_dict
    from ovg.benchmarker import BenchmarkRunner
    from ovg.data.datasets import BBBC021Dataset, Edges2ShoesDataset

    class LoRABenchmarkRunner(BenchmarkRunner):
        def _load_models(self):
            if not self.cfg.model_checkpoint.is_text_to_image:
                raise ValueError(
                    "This config evaluates existing SD1.5 text-conditioned adapters. "
                    "OVG class-conditioned evaluation needs its microscopy checkpoint "
                    "and an inversion adapter trained against that checkpoint."
                )
            super()._load_models()
            self.model_training_mode = "diffusion"
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.denoiser.eval().requires_grad_(False)
            self.vae.eval().requires_grad_(False)
            self.text_encoder.eval().requires_grad_(False)
            needs_adapter = any(
                item.name == "lora_inversion" and item.get("enabled", True)
                for item in self.cfg.benchmark.methods
            )
            if needs_adapter:
                state = torch.load(to_absolute_path(self.cfg.adapter_checkpoint),
                                   map_location="cpu", weights_only=True)
                if not state or any("branch" in key for key in state):
                    raise ValueError("Expected a single inversion adapter state dict")
                self.denoiser.add_adapter(LoraConfig(**OmegaConf.to_container(self.cfg.lora)),
                                          adapter_name="inversion")
                result = set_peft_model_state_dict(self.denoiser, state, adapter_name="inversion")
                missing = [key for key in result.missing_keys if "lora_" in key]
                if missing or result.unexpected_keys:
                    raise ValueError(f"Adapter mismatch: {missing=}, unexpected={result.unexpected_keys}")
                self.denoiser.disable_adapters()
            self._latent_cache = {}

        def _setup_dataset(self):
            frame = pd.read_csv(to_absolute_path(self.cfg.data.csv_path))
            edges = self.cfg.data.dataset == "edges2shoes"
            path_column = "image_path" if edges else "path"
            required = {"image_path", "image_id"} if edges else {"path", "class_idx", "res_class"}
            if not required.issubset(frame.columns) or frame.empty:
                raise ValueError(f"Nonempty BBBC021 CSV must contain {sorted(required)}")
            root = Path(to_absolute_path(self.cfg.data.image_root))
            paths = []
            for value in frame[path_column]:
                path = Path(value)
                path = path if path.is_absolute() else root / path
                with Image.open(path) as image:
                    if image.mode != "RGB":
                        raise ValueError(f"{path}: expected prepared RGB microscopy image; "
                                         "raw channel assembly/normalization must be specified")
                    if edges and image.size != (512, 256):
                        raise ValueError(f"Expected an edge/shoe pair of size 512x256: {path}")
                    image.verify()
                paths.append(str(path.resolve()))
            if len(set(paths)) != len(paths):
                raise ValueError("Duplicate images in test CSV would bias paired metrics")
            frame[path_column] = paths
            output = Path(self.cfg.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            resolved_csv = output / "resolved_test.csv"
            frame.to_csv(resolved_csv, index=False)
            options = OmegaConf.to_container(self.cfg.data)
            for key in ("dataset", "image_root", "num_classes", "csv_path"):
                options.pop(key)
            if edges:
                self.dataset = Edges2ShoesDataset(
                    csv_path=resolved_csv, root_dir=root, mode="translation",
                    img_size=self.cfg.data.hr_size, augmentation=False,
                    paired_mode=True, normalize=True)
            else:
                self.dataset = BBBC021Dataset(csv_path=resolved_csv, augmentation=False, **options)
            if self.num_samples > len(frame):
                raise ValueError("Requested more samples than the evaluation manifest contains")
            indices = torch.randperm(len(frame), generator=torch.Generator().manual_seed(
                self.cfg.get("sample_seed", self.seed)))
            indices = indices[:self.num_samples].tolist()
            frame.iloc[indices].assign(csv_row=indices).to_csv(output / "sample_manifest.csv", index=False)

        def _prepare_samples(self):
            # Hold images fixed across seeds; only stochastic model inputs vary.
            original_seed = self.seed
            try:
                self.seed = self.cfg.get("sample_seed", original_seed)
                return super()._prepare_samples()
            finally:
                self.seed = original_seed

        def _setup_metrics(self):
            super()._setup_metrics()
            if self.cfg.data.dataset == "edges2shoes":
                # The official paired archive supplies no biological/category labels.
                self.metrics = {k: v for k, v in self.metrics.items() if not k.startswith("knn")}

        def _compute_knn_recovery(self, *args, **kwargs):
            if self.cfg.data.dataset == "edges2shoes":
                return {}
            return super()._compute_knn_recovery(*args, **kwargs)

        def _setup_inversion_methods(self):
            supported = {"standard_inversion", "ours", "matched_ddim", "lora_inversion"}
            for item in self.cfg.benchmark.methods:
                if item.get("enabled", True) and item.name not in supported:
                    raise ValueError(f"Unsupported method: {item.name}")
            methods = super()._setup_inversion_methods()
            for index, item in enumerate(self.cfg.benchmark.methods):
                if item.get("enabled", True) and item.name in {"matched_ddim", "lora_inversion"}:
                    method = make_inversion_method(item.name == "lora_inversion")
                    method.name += f"_config_{index}"
                    methods.append(method)
            return methods

        def _get_embeddings(self, editing_mode, source_data, target_data):
            count = len(source_data)
            source, _ = self.encode_embeddings([self.cfg.conditioning.source_prompt] * count)
            target, uncond = self.encode_embeddings([self.cfg.conditioning.target_prompt] * count)
            return source, target, uncond

        @torch.no_grad()
        def _encode_latents(self, batch_input):
            # Upstream samples the VAE posterior separately for every method.
            # Reuse the identical sampled input for all compared inversions.
            key = hashlib.sha256(batch_input.cpu().numpy().tobytes()).hexdigest()
            if key not in self._latent_cache:
                self._latent_cache[key] = super()._encode_latents(batch_input).detach().cpu()
            return self._latent_cache[key].to(self.device, dtype=self.dtype).clone()

        # These upstream exploratory probes perform extra inversions with different
        # corruptions. They are not part of the requested SR comparison.
        def run_null_space_sensitivity_test(self):
            pass

        def run_differential_spectral_analysis(self):
            pass

        def run_blur_variance_analysis(self, sigma_range=None):
            pass

    return LoRABenchmarkRunner


@hydra.main(config_path="../../config", config_name="eval/bbbc021", version_base=None)
def main(cfg):
    import torch
    from accelerate.utils import set_seed

    if cfg.inference.guidance_scale != 1.0 or cfg.inference.eta != 0.0:
        raise ValueError("This initial OVG comparison requires deterministic CFG=1 sampling")
    if not Path(to_absolute_path(cfg.data.csv_path)).is_file():
        raise FileNotFoundError(
            f"Missing BBBC021 evaluation CSV: {cfg.data.csv_path}. "
            "See docs/bbbc021_superresolution.md for its schema and required artifacts."
        )
    os.environ.setdefault("WANDB_MODE", "disabled")
    load_upstream(Path(to_absolute_path(cfg.upstream_root)))
    set_seed(cfg.seed)
    runner = runner_class()(cfg)
    try:
        runner.setup()
        # Upstream includes Hydra's config name in HDF5 filenames. Our config
        # lives in a group (eval/bbbc021), so its filename has a parent folder.
        if cfg.benchmark.export_intermediates:
            from hydra.core.hydra_config import HydraConfig
            config_parent = Path(HydraConfig.get().job.config_name).parent
            (Path(cfg.benchmark.export_output_dir) / config_parent).mkdir(
                parents=True, exist_ok=True)
        results = runner.benchmark()
        summary = {
            "protocol": ("sd15_transfer_not_paper_microscopy_backbone" if cfg.data.dataset == "bbbc021"
                         else "sd15_transfer_not_paper_edges2shoes_backbone"),
            "upstream_revision": OVG_REVISION,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "num_samples": len(results["knn_labels"]),
            "metrics": {name: row["metrics"] for name, row in results["method_results"].items()},
        }
        output = Path(cfg.output_dir)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        torch.save(results, output / "visualization_samples.pt")
        if cfg.benchmark.export_intermediates:
            import runpy
            validator = Path(__file__).resolve().parents[2] / "scripts/analysis/validate_bbbc021.py"
            runpy.run_path(str(validator))["validate"](output)
        print(json.dumps(summary["metrics"], indent=2))
    finally:
        runner.run_logger.close()


if __name__ == "__main__":
    main()
