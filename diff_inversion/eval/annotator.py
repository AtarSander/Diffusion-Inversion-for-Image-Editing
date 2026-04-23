from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import PreTrainedModel, ViTForImageClassification, ViTImageProcessor


@dataclass
class Annotator:
    processor: Any
    model: PreTrainedModel


class GenerationsDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return T.ToPILImage()(make_grid(self.x[idx], nrow=1, normalize=True))


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_vit_annotator(model_name: str, device: str | torch.device = "auto") -> Annotator:
    device = resolve_device(device)
    logger.info("Loading ViT annotator '{}' on {}", model_name, device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    return Annotator(processor=processor, model=model)


def get_vit_cifar10_annotator(device: str | torch.device = "auto"):
    return get_vit_annotator("nateraw/vit-base-patch16-224-cifar10", device=device)


def get_vit_imagenet_annotator(device: str | torch.device = "auto"):
    return get_vit_annotator("google/vit-base-patch16-224", device=device)


def _get_model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def annotate(dataset, n_samples, batch_size, annotator: Annotator):
    labels = []
    device = _get_model_device(annotator.model)
    with torch.no_grad():
        for idx_start in tqdm(range(0, n_samples, batch_size)):
            idx_end = min(idx_start + batch_size, n_samples)
            dat_in = [dataset[idx] for idx in range(idx_start, idx_end)]
            inputs = annotator.processor(images=dat_in, return_tensors="pt").to(device)
            outputs = annotator.model(**inputs)
            logits = outputs.logits
            softmax_logits = F.softmax(logits, dim=1)
            _, max_index = torch.max(softmax_logits, dim=1)
            labels.append(max_index.cpu())
    return labels


def annotate_dl(dataloader, annotator: Annotator):
    labels = []
    device = _get_model_device(annotator.model)
    with torch.no_grad():
        for samples in tqdm(dataloader):
            inputs = annotator.processor(images=samples, return_tensors="pt").to(device)
            outputs = annotator.model(**inputs)
            logits = outputs.logits
            softmax_logits = F.softmax(logits, dim=1)
            _, max_index = torch.max(softmax_logits, dim=1)
            labels.append(max_index.cpu())
    return labels
