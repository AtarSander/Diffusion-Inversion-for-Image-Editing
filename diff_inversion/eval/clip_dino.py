from collections.abc import Sequence

import torch
from loguru import logger
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoProcessor

INTERPOLATION_MODES = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_clip(model_name: str, device: str | torch.device = "auto"):
    device = resolve_device(device)
    logger.info("Loading CLIP model '{}' on {}", model_name, device)
    clip_processor = AutoProcessor.from_pretrained(model_name)
    clip_model = AutoModel.from_pretrained(model_name).to(device)
    return clip_processor, clip_model


def output_to_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0, :]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")


def get_dino(
    model_name: str,
    device: str | torch.device = "auto",
    add_pooling_layer: bool = False,
):
    device = resolve_device(device)
    logger.info("Loading DINO model '{}' on {}", model_name, device)
    dino_model = AutoModel.from_pretrained(model_name, add_pooling_layer=add_pooling_layer).to(
        device
    )
    return dino_model


def get_clip_features(
    imgs: Sequence[Image.Image],
    clip_processor,
    clip_model,
    batch_size: int,
    device: str | torch.device = "auto",
):
    device = resolve_device(device)
    outs = []
    for batch_ids in range(0, len(imgs), batch_size):
        batch = imgs[batch_ids : batch_ids + batch_size]
        clip_batch_in = clip_processor(images=batch, return_tensors="pt").pixel_values.to(device)
        feats = output_to_feature_tensor(clip_model.get_image_features(clip_batch_in))
        outs.append(feats.detach().cpu())
    return torch.cat(outs)


def build_dino_transform(
    resize: int,
    center_crop: int,
    mean: Sequence[float],
    std: Sequence[float],
    interpolation: str = "bicubic",
):
    if interpolation not in INTERPOLATION_MODES:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=INTERPOLATION_MODES[interpolation]),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_dino_features(
    imgs: Sequence[Image.Image],
    dino_model,
    batch_size: int,
    transform,
    device: str | torch.device = "auto",
):
    device = resolve_device(device)
    outs = []
    for batch_ids in range(0, len(imgs), batch_size):
        batch = imgs[batch_ids : batch_ids + batch_size]
        pred_imgs_processed: torch.Tensor = torch.stack([transform(img) for img in batch])
        pred_imgs_processed = pred_imgs_processed.to(device)
        pred_features = dino_model(pred_imgs_processed).last_hidden_state[:, 0, :]
        outs.append(pred_features.detach().cpu())
    return torch.cat(outs)


def get_mean_cosine_sim(vec1, vec2):
    vec1 = vec1.view(vec1.shape[0], -1)
    vec2 = vec2.view(vec2.shape[0], -1)
    vec1 = torch.nn.functional.normalize(vec1, dim=1)
    vec2 = torch.nn.functional.normalize(vec2, dim=1)
    return torch.sum(vec1 * vec2, dim=1).mean()
