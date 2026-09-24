#!/usr/bin/env python3
"""Corrected reconstruction/background metrics for PIE-Bench editor panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pnp_inversion"))
from evaluation.matrics_calculator import LossG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-image", action="store_true")
    parser.add_argument("--panel-index", type=int, default=-1)
    parser.add_argument("--mask-erosion", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-stride", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=24)
    return parser.parse_args()


def decode_edit_mask(encoded: list[int]) -> np.ndarray:
    flat = np.zeros(512 * 512, dtype=bool)
    for start, count in zip(encoded[0::2], encoded[1::2]):
        flat[start : min(flat.size, start + count)] = True
    mask = flat.reshape(512, 512)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def load_rgb(path: Path, panel: bool = False, panel_index: int = -1) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if panel and image.width != image.height:
        panel_count = image.width // 512
        index = panel_index % panel_count
        left = index * 512
        image = image.crop((left, image.height - 512, left + 512, image.height))
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def tile_lpips(
    model: lpips.LPIPS,
    source: np.ndarray,
    output: np.ndarray,
    valid: np.ndarray,
    tile_size: int,
    stride: int,
    max_tiles: int,
    device: str,
) -> tuple[float, int]:
    locations = []
    for top in range(0, source.shape[0] - tile_size + 1, stride):
        for left in range(0, source.shape[1] - tile_size + 1, stride):
            if bool(valid[top : top + tile_size, left : left + tile_size].all()):
                locations.append((top, left))
    if not locations:
        return float("nan"), 0
    if len(locations) > max_tiles:
        indices = np.linspace(0, len(locations) - 1, max_tiles).round().astype(int)
        locations = [locations[index] for index in sorted(set(indices.tolist()))]
    source_tiles = np.stack(
        [source[top : top + tile_size, left : left + tile_size] for top, left in locations]
    )
    output_tiles = np.stack(
        [output[top : top + tile_size, left : left + tile_size] for top, left in locations]
    )
    source_tensor = torch.from_numpy(source_tiles).permute(0, 3, 1, 2).to(device) * 2 - 1
    output_tensor = torch.from_numpy(output_tiles).permute(0, 3, 1, 2).to(device) * 2 - 1
    with torch.no_grad():
        value = model(output_tensor, source_tensor, normalize=False).mean()
    return float(value.cpu()), len(locations)


def corrected_structure(loss: LossG, source: np.ndarray, output: np.ndarray, device: str) -> float:
    source_tensor = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(device)
    output_tensor = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(loss.calculate_global_ssim_loss(source_tensor, output_tensor).cpu())


def main() -> None:
    args = parse_args()
    methods = {}
    for value in args.method:
        name, separator, directory = value.partition("=")
        if not separator:
            raise ValueError(f"Expected NAME=DIR, got {value!r}")
        methods[name] = Path(directory)
    mapping = json.loads(args.mapping_file.read_text())
    lpips_model = lpips.LPIPS(net="squeeze", verbose=False).to(args.device).eval()
    dino = LossG(
        cfg={
            "dino_model_name": "dino_vitb8",
            "dino_global_patch_size": 224,
            "lambda_global_cls": 10.0,
            "lambda_global_ssim": 1.0,
            "lambda_global_identity": 1.0,
            "entire_A_every": 75,
            "lambda_entire_cls": 10,
            "lambda_entire_ssim": 1.0,
        },
        device=args.device,
    ).eval()
    rows = []
    for method, root in methods.items():
        for file_id, item in mapping.items():
            source = load_rgb(args.source_root / item["image_path"])
            output = load_rgb(root / item["image_path"], panel=True, panel_index=args.panel_index)
            valid = np.ones((512, 512), dtype=bool)
            if not args.full_image:
                valid = ~decode_edit_mask(item["mask"])
                if args.mask_erosion > 0:
                    valid = binary_erosion(valid, iterations=args.mask_erosion)
            if not bool(valid.any()):
                mse = psnr = ssim = tile_score = float("nan")
                tile_count = 0
            else:
                mse = float(np.square(output - source)[valid].mean())
                psnr = -10.0 * math.log10(max(mse, 1e-12))
                _, ssim_map = structural_similarity(
                    source, output, data_range=1.0, channel_axis=2,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    full=True,
                )
                if ssim_map.ndim == 3:
                    ssim_map = ssim_map.mean(axis=2)
                ssim = float(ssim_map[valid].mean())
                tile_score, tile_count = tile_lpips(
                    lpips_model, source, output, valid, args.tile_size,
                    args.tile_stride, args.max_tiles, args.device,
                )
            structure = corrected_structure(dino, source, output, args.device)
            row = {
                "method": method,
                "file_id": file_id,
                "valid_fraction": float(valid.mean()),
                "mse_valid": mse,
                "psnr_valid": psnr,
                "ssim_eroded": ssim,
                "lpips_valid_tiles": tile_score,
                "lpips_tile_count": tile_count,
                "structure_dino_scaled_correctly": structure,
            }
            rows.append(row)
            print(f"corrected metrics method={method} file_id={file_id}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    metric_names = [
        "mse_valid", "psnr_valid", "ssim_eroded", "lpips_valid_tiles",
        "structure_dino_scaled_correctly",
    ]
    summary = {"definitions": {
        "mse_valid": "MSE averaged only over valid pixels; lower is better",
        "psnr_valid": "PSNR derived from valid-pixel MSE; higher is better",
        "ssim_eroded": "SSIM map averaged at valid centres after mask erosion; higher is better",
        "lpips_valid_tiles": "LPIPS over tiles entirely inside the eroded valid region; lower is better",
        "structure_dino_scaled_correctly": "DINO self-similarity distance with RGB scaled to [0,1]; lower is better",
        "full_image": args.full_image,
        "mask_erosion": args.mask_erosion,
        "panel_index": args.panel_index,
    }, "methods": {}}
    for method, method_rows in grouped.items():
        summary["methods"][method] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in method_rows if math.isfinite(float(row[metric]))]
            summary["methods"][method][metric] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
            }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
