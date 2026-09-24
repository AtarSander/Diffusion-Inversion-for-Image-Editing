#!/usr/bin/env python3
"""Per-timestep base-vs-LoRA inversion diagnostics on PIE-Bench sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import run_editing_pnp as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--lora_checkpoint", type=Path, required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--baseline_metrics_root", type=Path, required=True)
    parser.add_argument("--experiment_metrics_root", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def tensor_output(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    sample = getattr(value, "sample", None)
    return sample if isinstance(sample, torch.Tensor) else None


def denoise_prediction(model, noisy, timestep, cond, captures):
    saved = {}
    handles = []
    modules = {
        "pnp_conv": model.unet.up_blocks[1].resnets[1],
        "self_attention": model.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1,
        "cross_attention": model.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2,
    }
    for name, module in modules.items():
        def hook(_module, _inputs, output, feature_name=name):
            tensor = tensor_output(output)
            if tensor is not None:
                saved[feature_name] = tensor.detach()
        handles.append(module.register_forward_hook(hook))
    try:
        eps = model.unet(noisy, timestep, encoder_hidden_states=cond).sample
    finally:
        for handle in handles:
            handle.remove()
    captures.update(saved)
    return eps


def predicted_clean(model, noisy, timestep, eps, index):
    timesteps = model.scheduler.timesteps
    alpha_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prev = (
        model.scheduler.alphas_cumprod[timesteps[index + 1]]
        if index < len(timesteps) - 1
        else model.scheduler.final_alpha_cumprod
    )
    mu = alpha_t.sqrt()
    sigma = (1 - alpha_t).sqrt()
    mu_prev = alpha_prev.sqrt()
    sigma_prev = (1 - alpha_prev).sqrt()
    pred_x0 = (noisy - sigma * eps) / mu
    return mu_prev * pred_x0 + sigma_prev * eps


def paired_step_metrics(model, base_trajectory, lora_trajectory, cond):
    rows = []
    for index, timestep in enumerate(model.scheduler.timesteps):
        base_noisy = base_trajectory[-1 - index]
        lora_noisy = lora_trajectory[-1 - index]
        base_features = {}
        base_eps = denoise_prediction(model, base_noisy, timestep, cond, base_features)
        lora_features = {}
        lora_eps = denoise_prediction(model, lora_noisy, timestep, cond, lora_features)
        base_offset = base_trajectory[-2 - index] - predicted_clean(
            model, base_noisy, timestep, base_eps, index
        )
        lora_offset = lora_trajectory[-2 - index] - predicted_clean(
            model, lora_noisy, timestep, lora_eps, index
        )
        feature_gaps = {
            name: float(
                (lora_features[name].float() - base_features[name].float())
                .square().mean().sqrt().cpu()
            )
            for name in base_features.keys() & lora_features.keys()
        }
        rows.append({
            "denoise_step": index,
            "timestep": int(timestep),
            "base_offset_rms": float(base_offset.float().square().mean().sqrt().cpu()),
            "lora_offset_rms": float(lora_offset.float().square().mean().sqrt().cpu()),
            **{f"{name}_feature_gap": value for name, value in feature_gaps.items()},
        })
        del base_features, lora_features, base_eps, lora_eps
    return rows


def load_metric_gaps(baseline_root: Path, experiment_root: Path):
    gaps = {}
    for editor in ("p2p", "pnp", "masactrl", "pix2pix-zero"):
        baseline_method = f"directinversion+{editor}"
        experiment_method = f"lora+directinversion+{editor}"
        with (baseline_root / f"{baseline_method}.csv").open(newline="") as handle:
            baseline = {row["file_id"]: row for row in csv.DictReader(handle)}
        with (experiment_root / f"{experiment_method}.csv").open(newline="") as handle:
            experiment = {row["file_id"]: row for row in csv.DictReader(handle)}
        for file_id in baseline.keys() & experiment.keys():
            record = gaps.setdefault(str(file_id), {})
            for metric in ("structure_distance", "mse_unedit_part", "lpips_unedit_part", "psnr_unedit_part", "ssim_unedit_part"):
                b = float(baseline[file_id][f"{baseline_method}|{metric}"])
                e = float(experiment[file_id][f"{experiment_method}|{metric}"])
                if math.isfinite(b) and math.isfinite(e):
                    record[f"{editor}:{metric}"] = b - e if metric in {"psnr_unedit_part", "ssim_unedit_part"} else e - b
    return gaps


def correlations(rows):
    predictors = (
        "endpoint_rms_base_to_lora", "mean_lora_offset_rms", "mean_offset_gap_rms",
        "lora_roundtrip_latent_rms", "pnp_conv_feature_gap", "self_attention_feature_gap",
        "cross_attention_feature_gap",
    )
    metric_names = sorted({name for row in rows for name in row.get("metric_gaps", {})})
    output = {}
    for predictor in predictors:
        output[predictor] = {}
        for metric in metric_names:
            pairs = [
                (float(row[predictor]), float(row["metric_gaps"][metric]))
                for row in rows if metric in row.get("metric_gaps", {})
                and math.isfinite(float(row[predictor]))
                and math.isfinite(float(row["metric_gaps"][metric]))
            ]
            if len(pairs) < 3:
                continue
            x, y = zip(*pairs)
            output[predictor][metric] = {
                "n": len(pairs),
                "pearson_r": float(pearsonr(x, y).statistic),
                "spearman_r": float(spearmanr(x, y).statistic),
            }
    return output


def main() -> None:
    args = parse_args()
    mapping = json.loads((args.data_path / "mapping_file.json").read_text())
    runner.model.load_lora(
        str(args.lora_checkpoint), rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    metric_gaps = load_metric_gaps(args.baseline_metrics_root, args.experiment_metrics_root)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    rows = []
    if args.output_jsonl.exists():
        for line in args.output_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add(str(row["file_id"]))

    selected = list(mapping.items())
    if args.max_images > 0:
        selected = selected[: args.max_images]
    for file_id, item in selected:
        if str(file_id) in completed:
            continue
        runner.setup_seed()
        image_path = args.data_path / "annotation_images" / item["image_path"]
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        base_trajectory, _, base_roundtrip = runner.model.extract_latents(
            50, str(image_path), inversion_prompt=prompt, use_lora_inversion=False,
        )
        lora_trajectory, _, lora_roundtrip = runner.model.extract_latents(
            50, str(image_path), inversion_prompt=prompt, use_lora_inversion=True,
        )
        _, cond = runner.model.get_text_embeds(prompt, "").chunk(2)
        runner.model.set_lora_enabled(False)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
            paired_steps = paired_step_metrics(
                runner.model, base_trajectory, lora_trajectory, cond
            )
        base_offsets = [step["base_offset_rms"] for step in paired_steps]
        lora_offsets = [step["lora_offset_rms"] for step in paired_steps]
        lora_roundtrip_errors = [
            float((actual.float() - expected.float()).square().mean().sqrt().cpu())
            for actual, expected in zip(lora_roundtrip, lora_trajectory[:-1])
        ]
        row = {
            "file_id": str(file_id),
            "image_path": item["image_path"],
            "endpoint_rms_base_to_lora": float(
                (lora_trajectory[-1].float() - base_trajectory[-1].float()).square().mean().sqrt().cpu()
            ),
            "mean_base_offset_rms": float(np.mean(base_offsets)),
            "mean_lora_offset_rms": float(np.mean(lora_offsets)),
            "mean_offset_gap_rms": float(np.mean(lora_offsets) - np.mean(base_offsets)),
            "lora_roundtrip_latent_rms": float(np.mean(lora_roundtrip_errors)),
            "pnp_conv_feature_gap": float(np.mean([step["pnp_conv_feature_gap"] for step in paired_steps])),
            "self_attention_feature_gap": float(np.mean([step["self_attention_feature_gap"] for step in paired_steps])),
            "cross_attention_feature_gap": float(np.mean([step["cross_attention_feature_gap"] for step in paired_steps])),
            "metric_gaps": metric_gaps.get(str(file_id), {}),
            "per_timestep": paired_steps,
        }
        with args.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        rows.append(row)
        print(f"trajectory diagnostics file_id={file_id}", flush=True)

    summary = {
        "num_images": len(rows),
        "checkpoint": str(args.lora_checkpoint),
        "correlations": correlations(rows),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
