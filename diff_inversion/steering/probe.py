# ABOUTME: Trains and applies a text-free WikiArt style probe: DINOv2 CLS features
# ABOUTME: plus multinomial logistic regression over the 27 style labels.
"""WikiArt style probe for scoring steering generations without CLIP text prompts.

Train:
    uv run python -m diff_inversion.steering.probe \
        --out_path output/steering/probe/wikiart_dinov2_probe.pt

The bundle stores the linear head, label names, backbone id and validation
accuracy; `probe_probs` applies it to PIL images at scoring time.
"""

import io
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch
from datasets import Image as ImageFeature
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel

DATASET = "huggan/wikiart"


@torch.no_grad()
def dino_features(
    images: list[Image.Image],
    processor,
    model,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Return L2-normalized CLS features, shape [len(images), hidden]."""
    outs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        pixels = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        features = model(pixel_values=pixels).last_hidden_state[:, 0, :]
        outs.append(torch.nn.functional.normalize(features, dim=-1).float().cpu())
    result = torch.cat(outs)
    assert result.shape == (len(images), model.config.hidden_size)
    return result


def load_probe(path: str | Path, device: str = "cuda"):
    """Load a probe bundle plus its frozen backbone and processor."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    processor = AutoImageProcessor.from_pretrained(bundle["backbone"])
    model = AutoModel.from_pretrained(bundle["backbone"]).to(device).eval()
    return bundle, processor, model


@torch.no_grad()
def probe_probs(
    images: list[Image.Image],
    bundle,
    processor,
    model,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Return [len(images), n_styles] probe probabilities."""
    features = dino_features(images, processor, model, batch_size, device)
    logits = features @ bundle["weight"].T + bundle["bias"]
    return logits.softmax(dim=-1)


def train(
    out_path: str,
    backbone: str = "facebook/dinov2-base",
    per_style: int = 200,
    scan_limit: int = 25000,
    seed: int = 7,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Train the style probe on a streamed WikiArt sample.

    Args:
        out_path: Destination .pt bundle (parent directories are created).
        backbone: Frozen feature extractor HF id.
        per_style: Image cap per style; rare styles keep whatever the scan finds.
        scan_limit: Maximum streamed rows, bounding download time.
        seed: Stream shuffle and split seed.
        val_fraction: Held-out fraction per style.
        batch_size: Images per backbone forward.
        device: Torch device for feature extraction.
    """
    load_dotenv()
    started = time.time()
    stream = load_dataset(DATASET, split="train", streaming=True)
    names = stream.features["style"].names
    stream = stream.cast_column("image", ImageFeature(decode=False))
    stream = stream.shuffle(seed=seed, buffer_size=2048)
    buckets: dict[int, list[Image.Image]] = defaultdict(list)
    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > scan_limit or all(
            len(buckets[idx]) >= per_style for idx in range(len(names))
        ):
            break
        style = row["style"]
        if len(buckets[style]) >= per_style:
            continue
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            buckets[style].append(image.convert("RGB"))
        if scanned % 2000 == 0:
            total = sum(len(v) for v in buckets.values())
            print(f"scanned {scanned}: kept {total}", flush=True)
    counts = {names[idx]: len(buckets[idx]) for idx in sorted(buckets)}
    if any(count < 10 for count in counts.values()) or len(counts) < len(names):
        raise RuntimeError(f"Too few images for some styles: {counts}")
    print(f"collected {sum(counts.values())} images over {len(counts)} styles", flush=True)
    processor = AutoImageProcessor.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device).eval()
    rng = np.random.default_rng(seed)
    splits = {"train": ([], []), "val": ([], [])}
    for style, images in buckets.items():
        order = rng.permutation(len(images))
        n_val = max(1, round(len(images) * val_fraction))
        for position, image_index in enumerate(order):
            split = "val" if position < n_val else "train"
            features, labels = splits[split]
            features.append(images[image_index])
            labels.append(style)
    bundles = {}
    for split, (images, labels) in splits.items():
        feats = dino_features(images, processor, model, batch_size, device)
        bundles[split] = (feats.numpy(), np.array(labels))
        print(f"{split}: {feats.shape}", flush=True)
    classifier = LogisticRegression(max_iter=5000, C=1.0)
    classifier.fit(*bundles["train"])
    val_features, val_labels = bundles["val"]
    predictions = classifier.predict(val_features)
    accuracy = float((predictions == val_labels).mean())
    per_class = {
        names[style]: float((predictions[val_labels == style] == style).mean())
        for style in sorted(set(val_labels))
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": torch.tensor(classifier.coef_, dtype=torch.float32),
            "bias": torch.tensor(classifier.intercept_, dtype=torch.float32),
            "labels": names,
            "backbone": backbone,
            "val_accuracy": accuracy,
            "val_accuracy_per_class": per_class,
            "class_counts": counts,
            "meta": {
                "dataset": DATASET,
                "per_style": per_style,
                "scan_limit": scan_limit,
                "rows_scanned": scanned,
                "seed": seed,
                "val_fraction": val_fraction,
                "git_sha": commit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_clock_seconds": round(time.time() - started, 1),
            },
        },
        out,
    )
    print(json.dumps({"val_accuracy": accuracy, **per_class}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    fire.Fire(train)
