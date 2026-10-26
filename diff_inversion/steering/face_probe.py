# ABOUTME: Trains a CelebA-HQ face-attribute probe: DINOv2 CLS features plus one
# ABOUTME: balanced logistic regression per attribute, applied to face crops at scoring.
"""CelebA-HQ attribute probe for scoring face steering generations.

Train (after unzipping CelebAMask-HQ.zip under data/raw/celebamask_hq):
    uv run python -m diff_inversion.steering.face_probe \
        --out_path output/steering/probe/celebahq_dinov2_probe.pt

The bundle stores one linear head per attribute; `face_probe_probs` applies
sigmoid per attribute, so probabilities are independent, not a softmax.
"""

import json
import subprocess
import time
from pathlib import Path

import fire
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoImageProcessor, AutoModel

from .probe import dino_features

CELEBAHQ_ROOT = Path("data/raw/celebamask_hq/CelebAMask-HQ")
ANNO_FILE = "CelebAMask-HQ-attribute-anno.txt"
IMG_DIR = "CelebA-HQ-img"


def load_anno(root: Path = CELEBAHQ_ROOT) -> tuple[list[str], list[dict]]:
    """Parse the attribute annotation file into (attribute names, per-image rows).

    Rows hold the image filename plus one +-1 flag per attribute.
    """
    lines = (root / ANNO_FILE).read_text().splitlines()
    count, names = int(lines[0]), lines[1].split()
    assert len(names) == 40, names
    rows = []
    for line in lines[2:]:
        parts = line.split()
        assert len(parts) == 41, line
        rows.append({"file": parts[0], **{n: int(v) for n, v in zip(names, parts[1:])}})
    assert len(rows) == count, (len(rows), count)
    return names, rows


def face_probe_probs(
    images: list[Image.Image],
    bundle,
    processor,
    model,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Return [len(images), n_attributes] independent sigmoid probabilities."""
    features = dino_features(images, processor, model, batch_size, device)
    logits = features @ bundle["weight"].T + bundle["bias"]
    assert logits.shape == (len(images), len(bundle["labels"]))
    return logits.sigmoid()


def train(
    out_path: str,
    backbone: str = "facebook/dinov2-base",
    sample_size: int = 10000,
    seed: int = 7,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Train one balanced logistic head per CelebA-HQ attribute on DINOv2 features.

    Args:
        out_path: Destination .pt bundle (parent directories are created).
        backbone: Frozen feature extractor HF id.
        sample_size: Images drawn from the 30k set (features computed once).
        seed: Sampling and split seed.
        val_fraction: Held-out fraction.
        batch_size: Images per backbone forward; files are read batch by batch.
        device: Torch device for feature extraction.
    """
    load_dotenv()
    started = time.time()
    names, rows = load_anno()
    rng = np.random.default_rng(seed)
    picked = [rows[i] for i in rng.permutation(len(rows))[:sample_size]]
    processor = AutoImageProcessor.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device).eval()
    feats = []
    for start in range(0, len(picked), batch_size):
        batch = []
        for row in picked[start : start + batch_size]:
            with Image.open(CELEBAHQ_ROOT / IMG_DIR / row["file"]) as image:
                batch.append(image.convert("RGB"))
        feats.append(dino_features(batch, processor, model, batch_size, device))
        if start % (batch_size * 50) == 0:
            print(f"featurized {start}/{len(picked)}", flush=True)
    features = torch.cat(feats).numpy()
    labels = np.array([[row[n] == 1 for n in names] for row in picked], dtype=np.int64)
    assert features.shape[0] == labels.shape[0] == sample_size
    n_val = round(sample_size * val_fraction)
    train_x, val_x = features[n_val:], features[:n_val]
    train_y, val_y = labels[n_val:], labels[:n_val]
    weights, biases, metrics = [], [], {}
    for index, name in enumerate(names):
        positives = int(train_y[:, index].sum())
        if positives < 20 or positives > len(train_y) - 20:
            raise RuntimeError(f"Attribute {name} too one-sided: {positives} positives")
        head = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        head.fit(train_x, train_y[:, index])
        weights.append(head.coef_[0])
        biases.append(head.intercept_[0])
        predictions = head.predict(val_x)
        metrics[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(val_y[:, index], predictions)),
            "accuracy": float((predictions == val_y[:, index]).mean()),
            "val_positive_rate": float(val_y[:, index].mean()),
        }
        print(f"{name}: bal_acc {metrics[name]['balanced_accuracy']:.3f}", flush=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": torch.tensor(np.stack(weights), dtype=torch.float32),
            "bias": torch.tensor(np.array(biases), dtype=torch.float32),
            "labels": names,
            "backbone": backbone,
            "val_metrics_per_attribute": metrics,
            "meta": {
                "dataset": "CelebAMask-HQ",
                "sample_size": sample_size,
                "seed": seed,
                "val_fraction": val_fraction,
                "git_sha": commit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_clock_seconds": round(time.time() - started, 1),
            },
        },
        out,
    )
    mean_bal = float(np.mean([m["balanced_accuracy"] for m in metrics.values()]))
    print(json.dumps({"mean_balanced_accuracy": mean_bal}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    fire.Fire(train)
