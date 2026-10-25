# ABOUTME: CSD style-descriptor model (Somepalli et al. 2024, arXiv:2404.01292): official
# ABOUTME: ViT-L checkpoint remapped onto HF CLIPVisionModel; embeds images for style similarity.
"""CSD style embeddings for steering evaluation.

Loads the authors' checkpoint (HF: tomg-group-umd/CSD-ViT-L), whose backbone is
the OpenAI-CLIP ViT-L/14 visual tower finetuned with a contrastive style loss.
The style embedding is ``normalize(pooler_output @ last_layer_style)``; style
similarity between two images is the cosine similarity of their embeddings.
"""

import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

CSD_REPO = "tomg-group-umd/CSD-ViT-L"
CLIP_VISION = "openai/clip-vit-large-patch14"


def _convert_backbone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename OpenAI-CLIP visual-tower keys to HF CLIPVisionModel vision_model keys."""
    renames = {
        "class_embedding": "embeddings.class_embedding",
        "positional_embedding": "embeddings.position_embedding.weight",
        "conv1.weight": "embeddings.patch_embedding.weight",
        "ln_pre.weight": "pre_layrnorm.weight",
        "ln_pre.bias": "pre_layrnorm.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
    }
    block_renames = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "attn.out_proj": "self_attn.out_proj",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
    }
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in renames:
            converted[renames[key]] = value
            continue
        prefix, block, rest = key.split(".", 2)[0], *key.split(".", 2)[1:]
        assert prefix == "transformer" and block == "resblocks", key
        index, rest = rest.split(".", 1)
        base = f"encoder.layers.{index}"
        if rest.startswith("attn.in_proj_"):
            kind = rest.removeprefix("attn.in_proj_")
            for name, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                converted[f"{base}.self_attn.{name}_proj.{kind}"] = chunk
            continue
        stem, kind = rest.rsplit(".", 1)
        converted[f"{base}.{block_renames[stem]}.{kind}"] = value
    return converted


class CSDModel(torch.nn.Module):
    """CLIP ViT-L visual tower with the CSD style projection head."""

    def __init__(self, vision: CLIPVisionModel, style_proj: torch.Tensor):
        super().__init__()
        self.vision = vision
        self.style_proj = torch.nn.Parameter(style_proj)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        assert pooled.shape == (pixel_values.shape[0], self.style_proj.shape[0])
        return torch.nn.functional.normalize(pooled @ self.style_proj, dim=-1)


def load_csd(device: str) -> tuple[CLIPImageProcessor, CSDModel]:
    """Download the CSD checkpoint and return (image processor, eval-mode model)."""
    path = hf_hub_download(CSD_REPO, "pytorch_model.bin")
    allow = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float64DType,
        argparse.Namespace,
    ]
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()
    }
    style_proj = state.pop("last_layer_style")
    state.pop("last_layer_content")
    backbone = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    assert len(backbone) == len(state), "Unexpected non-backbone keys in checkpoint"
    vision = CLIPVisionModel.from_pretrained(CLIP_VISION)
    vision.load_state_dict(_convert_backbone(backbone), strict=True)
    model = CSDModel(vision, style_proj).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(CLIP_VISION)
    return processor, model


@torch.no_grad()
def csd_style_features(
    images: list[Image.Image],
    processor: CLIPImageProcessor,
    model: CSDModel,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed images into L2-normalized CSD style descriptors, shape [N, 768]."""
    features = []
    for start in range(0, len(images), batch_size):
        batch = processor(images=images[start : start + batch_size], return_tensors="pt").to(
            device
        )
        features.append(model(batch["pixel_values"]).float().cpu())
    return torch.cat(features)


def mean_ref_similarity(features: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity of each (normalized) feature to a normalized ref set."""
    assert features.shape[-1] == refs.shape[-1]
    return (features @ refs.T).mean(dim=-1)
