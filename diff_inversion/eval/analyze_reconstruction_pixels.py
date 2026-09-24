#!/usr/bin/env python3
"""Paired, pixel-level reconstruction diagnostics for PIE-Bench outputs.

This intentionally complements (rather than replaces) the official evaluator.  It
recomputes the official masked MSE, a mask-area-normalized MSE/PSNR, and breaks
the source/output error down by distance from the edit mask and source-image
texture.  The two methods are always paired on the same PIE-Bench example.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


EDITORS = ("p2p", "pnp", "masactrl", "pix2pix-zero")
RINGS = ("boundary_0_4", "near_5_16", "mid_17_32", "far_gt_32")
TEXTURES = ("smooth", "medium", "edge")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    storage = (repo / "checkpoints").resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=repo / "data/raw/PIE-Bench_v1/mapping_file.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo / "data/raw/PIE-Bench_v1/annotation_images",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=storage / "evaluation/sd15_fp16_baselines",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            storage
            / "evaluation/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors"
        ),
    )
    parser.add_argument("--baseline-prefix", default="directinversion")
    parser.add_argument("--experiment-prefix", default="lora+directinversion")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decode_edit_mask(encoded: list[int]) -> np.ndarray:
    flat = np.zeros(512 * 512, dtype=bool)
    for start, count in zip(encoded[0::2], encoded[1::2]):
        flat[start : min(flat.size, start + count)] = True
    mask = flat.reshape(512, 512)
    # Match the official evaluator's annotation-boundary workaround.
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def box_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean square dilation without scipy."""
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask.astype(np.int32), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    width = 2 * radius + 1
    window_sum = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return window_sum > 0


def region_masks(edit_mask: np.ndarray, source: np.ndarray) -> tuple[dict, dict]:
    unedited = ~edit_mask
    d4 = box_dilate(edit_mask, 4)
    d16 = box_dilate(edit_mask, 16)
    d32 = box_dilate(edit_mask, 32)
    rings = {
        "boundary_0_4": unedited & d4,
        "near_5_16": unedited & d16 & ~d4,
        "mid_17_32": unedited & d32 & ~d16,
        "far_gt_32": unedited & ~d32,
    }

    rgb = source.astype(np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gy, gx = np.gradient(luminance)
    gradient = np.hypot(gx, gy)
    q1, q2 = np.quantile(gradient[unedited], (1 / 3, 2 / 3))
    textures = {
        "smooth": unedited & (gradient <= q1),
        "medium": unedited & (gradient > q1) & (gradient <= q2),
        "edge": unedited & (gradient > q2),
    }
    return rings, textures


def load_evaluator_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width != height:
        image = image.crop((width - 512, height - 512, width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (512, 512, 3):
        raise ValueError(f"Expected 512x512 RGB after evaluator crop, got {array.shape}: {path}")
    return array


def source_group(image_path: str) -> str:
    parts = Path(image_path).parts
    if "1_artificial" in parts:
        return "artificial_graphic"
    if "2_natural" in parts:
        return "photographic"
    return "random_unknown"


def category_name(category_id: str) -> str:
    return {
        "0": "random",
        "1": "change_object",
        "2": "add_object",
        "3": "delete_object",
        "4": "change_content",
        "5": "change_pose",
        "6": "change_color",
        "7": "change_material",
        "8": "change_background",
        "9": "change_style",
    }[category_id]


def new_region_accumulator() -> dict:
    return {
        "pixels": 0,
        "baseline_sse": 0.0,
        "experiment_sse": 0.0,
        "signed_excess_sse": 0.0,
        "positive_excess_sse": 0.0,
    }


def update_region(acc: dict, region: np.ndarray, base_sq: np.ndarray, exp_sq: np.ndarray) -> None:
    count = int(region.sum())
    if count == 0:
        return
    # Each selected pixel contributes three color-channel errors.
    b = base_sq[region]
    e = exp_sq[region]
    delta = e - b
    acc["pixels"] += count
    acc["baseline_sse"] += float(b.sum())
    acc["experiment_sse"] += float(e.sum())
    acc["signed_excess_sse"] += float(delta.sum())
    acc["positive_excess_sse"] += float(np.clip(delta, 0, None).sum())


def finalize_regions(accumulators: dict[str, dict]) -> dict:
    total_positive = sum(x["positive_excess_sse"] for x in accumulators.values())
    result = {}
    for name, acc in accumulators.items():
        channels = 3 * acc["pixels"]
        result[name] = {
            "pixels": acc["pixels"],
            "baseline_mse_x10000": 10000 * acc["baseline_sse"] / channels if channels else None,
            "experiment_mse_x10000": 10000 * acc["experiment_sse"] / channels if channels else None,
            "signed_gap_mse_x10000": 10000 * acc["signed_excess_sse"] / channels if channels else None,
            "share_of_positive_excess_sse": (
                acc["positive_excess_sse"] / total_positive if total_positive else None
            ),
        }
    return result


def read_official_mse(path: Path, method: str) -> dict[str, float]:
    key = f"{method}|mse_unedit_part"
    with path.open(newline="") as handle:
        return {
            row["file_id"]: float(row[key])
            for row in csv.DictReader(handle)
            if row[key].strip().lower() != "nan"
        }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping_file.read_text())
    output = {
        "definitions": {
            "gap_direction": "experiment minus baseline; positive is worse",
            "ring_distance": "Chebyshev pixel distance on the unedited side of the PIE mask",
            "texture": "within-image terciles of source luminance gradient among unedited pixels",
        },
        "editors": {},
    }

    for editor in EDITORS:
        base_method = f"{args.baseline_prefix}+{editor}"
        exp_method = f"{args.experiment_prefix}+{editor}"
        base_root = args.baseline_root / base_method / "annotation_images"
        exp_root = args.experiment_root / exp_method / "annotation_images"
        base_csv = read_official_mse(
            args.baseline_root / "metrics" / f"{base_method}.csv", base_method
        )
        exp_csv = read_official_mse(
            args.experiment_root / "metrics" / f"{exp_method}.csv", exp_method
        )

        rings_acc = {name: new_region_accumulator() for name in RINGS}
        texture_acc = {name: new_region_accumulator() for name in TEXTURES}
        group_acc = defaultdict(new_region_accumulator)
        category_acc = defaultdict(new_region_accumulator)
        grid_b_sse = np.zeros((8, 8), dtype=np.float64)
        grid_e_sse = np.zeros((8, 8), dtype=np.float64)
        grid_pixels = np.zeros((8, 8), dtype=np.int64)
        official_recompute_errors = []
        per_image = []

        for file_id, item in mapping.items():
            edit_mask = decode_edit_mask(item["mask"])
            unedited = ~edit_mask
            q = float(unedited.mean())
            if q == 0:
                continue

            relative = Path(item["image_path"])
            source = load_evaluator_image(args.source_root / relative)
            baseline = load_evaluator_image(base_root / relative)
            experiment = load_evaluator_image(exp_root / relative)
            base_error = baseline - source
            exp_error = experiment - source
            base_sq = base_error**2
            exp_sq = exp_error**2
            rings, textures = region_masks(edit_mask, source)

            for name, region in rings.items():
                update_region(rings_acc[name], region, base_sq, exp_sq)
            for name, region in textures.items():
                update_region(texture_acc[name], region, base_sq, exp_sq)
            update_region(group_acc[source_group(item["image_path"])], unedited, base_sq, exp_sq)
            update_region(category_acc[category_name(item["editing_type_id"])], unedited, base_sq, exp_sq)

            for gy in range(8):
                for gx in range(8):
                    ys, xs = slice(gy * 64, (gy + 1) * 64), slice(gx * 64, (gx + 1) * 64)
                    region = unedited[ys, xs]
                    count = int(region.sum())
                    if count:
                        grid_pixels[gy, gx] += count
                        grid_b_sse[gy, gx] += float(base_sq[ys, xs][region].sum())
                        grid_e_sse[gy, gx] += float(exp_sq[ys, xs][region].sum())

            b_mse = float(base_sq[unedited].mean())
            e_mse = float(exp_sq[unedited].mean())
            b_reported = q * b_mse
            e_reported = q * e_mse
            official_recompute_errors.extend(
                (
                    abs(b_reported - base_csv[file_id]),
                    abs(e_reported - exp_csv[file_id]),
                )
            )

            base_bias = base_error[unedited].mean(axis=0)
            exp_bias = exp_error[unedited].mean(axis=0)
            base_bias_mse = float(np.mean(base_bias**2))
            exp_bias_mse = float(np.mean(exp_bias**2))
            per_image.append(
                {
                    "file_id": file_id,
                    "image_path": item["image_path"],
                    "category": category_name(item["editing_type_id"]),
                    "source_group": source_group(item["image_path"]),
                    "original_prompt": item["original_prompt"],
                    "editing_prompt": item["editing_prompt"],
                    "instruction": item["editing_instruction"],
                    "unedited_fraction": q,
                    "baseline_normalized_mse_x10000": 10000 * b_mse,
                    "experiment_normalized_mse_x10000": 10000 * e_mse,
                    "normalized_mse_gap_x10000": 10000 * (e_mse - b_mse),
                    "baseline_normalized_psnr": -10 * math.log10(b_mse),
                    "experiment_normalized_psnr": -10 * math.log10(e_mse),
                    "normalized_psnr_worsening": 10 * math.log10(e_mse / b_mse),
                    "baseline_rgb_bias": base_bias.tolist(),
                    "experiment_rgb_bias": exp_bias.tolist(),
                    "baseline_bias_mse_x10000": 10000 * base_bias_mse,
                    "experiment_bias_mse_x10000": 10000 * exp_bias_mse,
                    "bias_mse_gap_x10000": 10000 * (exp_bias_mse - base_bias_mse),
                }
            )

        b_norm = [x["baseline_normalized_mse_x10000"] for x in per_image]
        e_norm = [x["experiment_normalized_mse_x10000"] for x in per_image]
        b_psnr = [x["baseline_normalized_psnr"] for x in per_image]
        e_psnr = [x["experiment_normalized_psnr"] for x in per_image]
        b_bias = [x["baseline_bias_mse_x10000"] for x in per_image]
        e_bias = [x["experiment_bias_mse_x10000"] for x in per_image]
        gaps = np.array([x["normalized_mse_gap_x10000"] for x in per_image])
        positive = np.clip(gaps, 0, None)
        order = np.argsort(gaps)[::-1]
        top_n = max(1, math.ceil(len(gaps) * 0.10))

        grid_denominator = 3 * grid_pixels
        grid_gap = np.divide(
            10000 * (grid_e_sse - grid_b_sse),
            grid_denominator,
            out=np.full((8, 8), np.nan),
            where=grid_denominator > 0,
        )
        output["editors"][editor] = {
            "n": len(per_image),
            "official_mse_recompute_max_abs_error": max(official_recompute_errors),
            "normalized_mask_metrics": {
                "baseline_mse_x10000": mean(b_norm),
                "experiment_mse_x10000": mean(e_norm),
                "mse_gap_x10000": mean([e - b for b, e in zip(b_norm, e_norm)]),
                "baseline_psnr": mean(b_psnr),
                "experiment_psnr": mean(e_psnr),
                "psnr_worsening": mean([b - e for b, e in zip(b_psnr, e_psnr)]),
            },
            "global_rgb_bias_component": {
                "baseline_bias_mse_x10000": mean(b_bias),
                "experiment_bias_mse_x10000": mean(e_bias),
                "bias_mse_gap_x10000": mean([e - b for b, e in zip(b_bias, e_bias)]),
                "share_of_total_signed_mse_gap": (
                    sum(e - b for b, e in zip(b_bias, e_bias)) / sum(e - b for b, e in zip(b_norm, e_norm))
                    if sum(e - b for b, e in zip(b_norm, e_norm)) != 0
                    else None
                ),
            },
            "by_mask_distance": finalize_regions(rings_acc),
            "by_source_texture": finalize_regions(texture_acc),
            "by_source_group": finalize_regions(dict(group_acc)),
            "by_category": finalize_regions(dict(category_acc)),
            "grid_signed_gap_mse_x10000": grid_gap.tolist(),
            "normalized_mse_outlier_concentration": {
                "top_10pct_share_of_positive_per_image_gap": (
                    float(positive[order[:top_n]].sum() / positive.sum()) if positive.sum() else None
                )
            },
            "worst_normalized_mse": [per_image[i] for i in order[:20]],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({editor: data["normalized_mask_metrics"] for editor, data in output["editors"].items()}, indent=2))


if __name__ == "__main__":
    main()
