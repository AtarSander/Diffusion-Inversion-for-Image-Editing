#!/usr/bin/env python3
"""Cross Pix2Pix-Zero endpoint, reference-attention, and offset sources."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import run_editing_pix2pix_zero as runner
from utils.utils import txt_draw

LABELS = ("base", "lora")


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
    parser.add_argument("--rerun_exist_images", action="store_true")
    return parser.parse_args()


def method_name(endpoint: str, attention: str, offsets: str) -> str:
    return f"pix2pix-cross_e-{endpoint}_a-{attention}_o-{offsets}"


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    runner.pipe, runner.edit_pipe = runner.load_pipelines(args.model_key)
    runner.load_lora_inversion(
        str(args.lora_checkpoint), rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    mapping = json.loads((args.data_path / "mapping_file.json").read_text())
    diagnostics = {}

    for file_id, item in mapping.items():
        image_path = args.data_path / "annotation_images" / item["image_path"]
        image_gt = Image.open(image_path).convert("RGB").resize(
            (512, 512), Image.Resampling.LANCZOS
        )
        prompt_src = item["original_prompt"].replace("[", "").replace("]", "")
        prompt_tar = item["editing_prompt"].replace("[", "").replace("]", "")
        runner.setup_seed()
        prompt_str = runner.generate_caption(image_gt)

        trajectories = {}
        runner.set_lora_enabled(False)
        trajectories["base"], _, _ = runner.pipe(
            prompt_str, guidance_scale=args.inversion_guidance_scale,
            num_inversion_steps=runner.NUM_DDIM_STEPS, img=image_gt,
        )
        runner.set_lora_enabled(True)
        try:
            trajectories["lora"], _, _ = runner.pipe(
                prompt_str, guidance_scale=args.inversion_guidance_scale,
                num_inversion_steps=runner.NUM_DDIM_STEPS, img=image_gt,
            )
        finally:
            runner.set_lora_enabled(False)

        endpoints = {name: values[-1].detach() for name, values in trajectories.items()}
        mean_src = runner.load_sentence_embeddings(
            [prompt_src], runner.edit_pipe.tokenizer,
            runner.edit_pipe.text_encoder, device=runner.device,
        )
        mean_tar = runner.load_sentence_embeddings(
            [prompt_tar], runner.edit_pipe.tokenizer,
            runner.edit_pipe.text_encoder, device=runner.device,
        )
        edit_dir = (mean_tar.mean(0) - mean_src.mean(0)).unsqueeze(0)

        attention_maps = {}
        offsets = {}
        reconstructions = {}
        for label in LABELS:
            reconstruction, attention_maps[label], offsets[label] = runner.edit_pipe(
                prompt_str,
                num_inference_steps=runner.NUM_DDIM_STEPS,
                x_in=endpoints[label],
                edit_dir=edit_dir,
                guidance_amount=runner.XA_GUIDANCE,
                guidance_scale=7.5,
                negative_prompt=prompt_str,
                latent_list=trajectories[label],
                only_sample=True,
                return_reference_artifacts=True,
            )
            reconstructions[label] = reconstruction[0]

        instruction = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )
        for endpoint_label, attention_label, offset_label in itertools.product(
            LABELS, repeat=3
        ):
            name = method_name(endpoint_label, attention_label, offset_label)
            destination = args.output_path / name / "annotation_images" / item["image_path"]
            if destination.exists() and not args.rerun_exist_images:
                continue
            _, edited = runner.edit_pipe(
                prompt_str,
                num_inference_steps=runner.NUM_DDIM_STEPS,
                x_in=endpoints[endpoint_label],
                edit_dir=edit_dir,
                guidance_amount=runner.XA_GUIDANCE,
                guidance_scale=7.5,
                negative_prompt=prompt_str,
                reference_attention_maps=attention_maps[attention_label],
                noise_loss_list_override=offsets[offset_label],
            )
            panel = np.concatenate(
                (
                    instruction,
                    image_array(image_gt),
                    image_array(reconstructions[endpoint_label]),
                    image_array(edited[0]),
                ),
                axis=1,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(destination)

        diagnostics[str(file_id)] = {
            "image_path": item["image_path"],
            "caption": prompt_str,
            "endpoint_rms_base_to_lora": float(
                (endpoints["lora"].float() - endpoints["base"].float())
                .square().mean().sqrt().cpu()
            ),
            "offset_rms": {
                label: [float(value.float().square().mean().sqrt().cpu()) for value in offsets[label]]
                for label in LABELS
            },
        }
        (args.output_path / "crossing_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"completed crossing file_id={file_id}", flush=True)


if __name__ == "__main__":
    main()
