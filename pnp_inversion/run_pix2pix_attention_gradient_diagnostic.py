#!/usr/bin/env python3
"""Measure Pix2Pix-Zero attention-loss and latent-gradient scales."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import run_editing_pix2pix_zero as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--lora_checkpoint", type=Path, required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--model_key", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inversion_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--normalized_loss_scales", nargs="+", type=float,
                        default=[1.0, 1e5, 1e6, 1e7, 1e8])
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_records(record_root: Path) -> dict:
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(record_root.glob("*/*.json"))
    ]
    summary: dict[str, dict] = {}
    for mode in ("base", "lora"):
        selected = [payload for payload in payloads if payload["mode"] == mode]
        records = [record for payload in selected for record in payload["records"]]
        grad_ratios = [
            record["legacy_grad_norm"] / record["normalized_grad_norm"]
            for record in records
            if record["normalized_grad_norm"] > 0.0
        ]
        loss_ratios = [
            record["legacy_loss"] / record["normalized_loss"]
            for record in records
            if record["normalized_loss"] > 0.0
        ]
        layer_fractions: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"legacy": [], "normalized": []}
        )
        for record in records:
            legacy_total = sum(layer["legacy_loss"] for layer in record["layers"])
            normalized_total = sum(layer["mean_loss"] for layer in record["layers"])
            for layer in record["layers"]:
                if legacy_total > 0.0:
                    layer_fractions[layer["name"]]["legacy"].append(
                        layer["legacy_loss"] / legacy_total
                    )
                if normalized_total > 0.0:
                    layer_fractions[layer["name"]]["normalized"].append(
                        layer["mean_loss"] / normalized_total
                    )
        layers = {
            name: {
                "legacy_fraction": numeric_summary(values["legacy"]),
                "normalized_fraction": numeric_summary(values["normalized"]),
            }
            for name, values in sorted(layer_fractions.items())
        }
        ratio_summary = numeric_summary(grad_ratios)
        ratio_median = ratio_summary["median"]
        scaled_gradient_stats = {}
        scale_keys = sorted({scale_key for record in records for scale_key in record.get("normalized_scaled_gradients", {})}, key=float)
        for scale_key in scale_keys:
            scale_records = [record["normalized_scaled_gradients"][scale_key] for record in records if scale_key in record.get("normalized_scaled_gradients", {})]
            norms = [entry["grad_norm"] for entry in scale_records]
            scaled_gradient_stats[scale_key] = {
                "grad_norm": numeric_summary(norms),
                "cosine_to_legacy": numeric_summary([entry["cosine_to_legacy"] for entry in scale_records]),
                "fraction_above_1": sum(value > 1.0 for value in norms) / len(norms) if norms else None,
                "fraction_nonzero": sum(value > 0.0 for value in norms) / len(norms) if norms else None,
            }
        summary[mode] = {
            "images": len(selected),
            "steps": len(records),
            "legacy_loss": numeric_summary(
                [record["legacy_loss"] for record in records]
            ),
            "normalized_loss": numeric_summary(
                [record["normalized_loss"] for record in records]
            ),
            "legacy_to_normalized_loss_ratio": numeric_summary(loss_ratios),
            "legacy_grad_norm": numeric_summary(
                [record["legacy_grad_norm"] for record in records]
            ),
            "normalized_grad_norm": numeric_summary(
                [record["normalized_grad_norm"] for record in records]
            ),
            "legacy_to_normalized_grad_ratio": ratio_summary,
            "gradient_cosine": numeric_summary(
                [record["gradient_cosine"] for record in records]
            ),
            "legacy_fraction_clipped_at_1": (
                sum(record["legacy_grad_norm"] > 1.0 for record in records)
                / len(records)
                if records
                else None
            ),
            "normalized_fraction_clipped_at_1": (
                sum(record["normalized_grad_norm"] > 1.0 for record in records)
                / len(records)
                if records
                else None
            ),
            "normalized_guidance_matching_legacy_0.1": (
                0.1 * ratio_median if ratio_median is not None else None
            ),
            "normalized_scaled_gradients": scaled_gradient_stats,
            "layers": layers,
        }
    return {
        "objective": (
            "Compare legacy summed and equal-per-layer-mean attention gradients "
            "on identical edit-loop states. The actual diagnostic path follows "
            "the unclipped legacy objective."
        ),
        "modes": summary,
    }


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max_images must be positive")
    if any(not np.isfinite(scale) or scale <= 0 for scale in args.normalized_loss_scales):
        raise ValueError("--normalized_loss_scales must contain only finite positive values")
    mapping_path = args.data_path / "mapping_file.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    args.output_path.mkdir(parents=True, exist_ok=True)
    record_root = args.output_path / "records"

    runner.pipe, runner.edit_pipe = runner.load_pipelines(args.model_key)
    runner.load_lora_inversion(
        str(args.lora_checkpoint),
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        scale=args.lora_scale,
    )

    for image_index, (file_id, item) in enumerate(mapping.items()):
        if args.max_images is not None and image_index >= args.max_images:
            break
        safe_id = str(file_id).replace("/", "_")
        record_paths = {
            mode: record_root / mode / f"{safe_id}.json"
            for mode in ("base", "lora")
        }
        if not args.rerun and all(path.exists() for path in record_paths.values()):
            print(f"skip completed file_id={file_id}", flush=True)
            continue

        image_path = args.data_path / "annotation_images" / item["image_path"]
        image_gt = Image.open(image_path).convert("RGB").resize(
            (512, 512), Image.Resampling.LANCZOS
        )
        prompt_src = item["original_prompt"].replace("[", "").replace("]", "")
        prompt_tar = item["editing_prompt"].replace("[", "").replace("]", "")
        runner.setup_seed()
        caption = runner.generate_caption(image_gt)

        trajectories = {}
        runner.set_lora_enabled(False)
        trajectories["base"], _, _ = runner.pipe(
            caption,
            guidance_scale=args.inversion_guidance_scale,
            num_inversion_steps=runner.NUM_DDIM_STEPS,
            img=image_gt,
        )
        runner.set_lora_enabled(True)
        try:
            trajectories["lora"], _, _ = runner.pipe(
                caption,
                guidance_scale=args.inversion_guidance_scale,
                num_inversion_steps=runner.NUM_DDIM_STEPS,
                img=image_gt,
            )
        finally:
            runner.set_lora_enabled(False)

        mean_src = runner.load_sentence_embeddings(
            [prompt_src], runner.edit_pipe.tokenizer,
            runner.edit_pipe.text_encoder, device=runner.device,
        )
        mean_tar = runner.load_sentence_embeddings(
            [prompt_tar], runner.edit_pipe.tokenizer,
            runner.edit_pipe.text_encoder, device=runner.device,
        )
        edit_direction = (mean_tar.mean(0) - mean_src.mean(0)).unsqueeze(0)

        for mode in ("base", "lora"):
            record_path = record_paths[mode]
            if record_path.exists() and not args.rerun:
                continue
            trajectory = trajectories[mode]
            records: list[dict] = []
            reconstruction, edited = runner.edit_pipe(
                caption,
                num_inference_steps=runner.NUM_DDIM_STEPS,
                x_in=trajectory[-1].detach(),
                edit_dir=edit_direction,
                guidance_amount=runner.XA_GUIDANCE,
                attention_loss_reduction="legacy_sum",
                attention_grad_clip_norm=None,
                guidance_scale=7.5,
                negative_prompt=caption,
                latent_list=trajectory,
                attention_gradient_diagnostics=records,
                diagnostic_guidance_amounts=(0.025, 0.05, 0.075, 0.1),
                diagnostic_clip_norm=1.0,
                diagnostic_normalized_loss_scales=tuple(args.normalized_loss_scales),
            )
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                json.dumps(
                    {
                        "file_id": str(file_id),
                        "image_path": item["image_path"],
                        "mode": mode,
                        "caption": caption,
                        "inversion_guidance_scale": args.inversion_guidance_scale,
                        "records": records,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            panel = np.concatenate(
                (
                    image_array(image_gt),
                    image_array(reconstruction[0]),
                    image_array(edited[0]),
                ),
                axis=1,
            )
            destination = (
                args.output_path / mode / "annotation_images" / item["image_path"]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(destination)
            print(
                f"completed file_id={file_id} mode={mode} steps={len(records)}",
                flush=True,
            )

        summary = summarize_records(record_root)
        (args.output_path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"saved diagnostic summary to {args.output_path / 'summary.json'}")


if __name__ == "__main__":
    main()
