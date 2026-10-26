# ABOUTME: Scores face-attribute steering runs: CLIP prompt fidelity and zero-shot
# ABOUTME: attribute prob on full images; probe P(attr) and identity sim on MTCNN crops.
"""Face-attribute steering scores.

    uv run python -m diff_inversion.steering.score_faces \
        --run_dir output/steering/celeba_smiling/<ts>/generate \
        --target_attribute Smiling \
        --probe_path output/steering/probe/celebahq_dinov2_probe.pt

Per image: p_target (CLIP zero-shot attribute probability), clip_t (prompt
fidelity), p_probe (DINOv2 attribute probe on the face crop), id_sim (FaceNet
cosine to the base image of the same prompt/seed; 1.0 on base rows). Images
where no face is detected keep NaN for crop metrics and are dropped pairwise
in the aggregation; the miss rate is printed and stored.
"""

import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from PIL import Image

from diff_inversion.eval.clip_dino import (
    get_clip,
    get_clip_features,
    output_to_feature_tensor,
)

from .face_probe import face_probe_probs
from .probe import load_probe
from .score import sem, variant_key

ATTRIBUTE_PROMPTS = {
    "Smiling": ("a photo of a smiling person", "a photo of a person who is not smiling"),
    "Eyeglasses": (
        "a photo of a person wearing glasses",
        "a photo of a person without glasses",
    ),
    "Young": ("a photo of a young person", "a photo of an old person"),
}
CROP_MARGIN = 0.35
METRICS = ("p_target", "clip_t", "p_probe", "id_sim")


def face_box(detector: MTCNN, image: Image.Image) -> list[float] | None:
    """Largest detected face box, or None."""
    boxes, _ = detector.detect(image)
    if boxes is None or not len(boxes):
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return list(boxes[max(range(len(boxes)), key=areas.__getitem__)])


def probe_crop(image: Image.Image, box: list[float]) -> Image.Image:
    """Square crop around the box with margin, CelebA-HQ-like framing."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + CROP_MARGIN) / 2
    left, top = max(0, int(cx - half)), max(0, int(cy - half))
    right = min(image.width, int(cx + half))
    bottom = min(image.height, int(cy + half))
    return image.crop((left, top, right, bottom))


def aggregate_faces(rows: list[dict]) -> list[dict]:
    """Variant summaries with paired deltas vs base; NaN pairs are dropped per metric."""
    base = {(row["prompt"], row["seed"]): row for row in rows if row["method"] == "base"}
    if len(base) != sum(row["method"] == "base" for row in rows):
        raise ValueError("Expected exactly one base image per (prompt, seed)")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(variant_key(row), []).append(row)
    summaries = []
    for key, members in groups.items():
        summary: dict = {"variant": key, "n": len(members)}
        for metric in METRICS:
            pairs = [
                (row[metric], base[(row["prompt"], row["seed"])][metric])
                for row in members
                if not math.isnan(row[metric])
                and not math.isnan(base[(row["prompt"], row["seed"])][metric])
            ]
            summary[f"{metric}_n"] = len(pairs)
            if not pairs:
                summary[f"{metric}_mean"] = summary[f"{metric}_sem"] = float("nan")
                summary[f"delta_{metric}_mean"] = summary[f"delta_{metric}_sem"] = float("nan")
                continue
            values = [value for value, _ in pairs]
            deltas = [value - base_value for value, base_value in pairs]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sem"] = sem(values)
            summary[f"delta_{metric}_mean"] = statistics.mean(deltas)
            summary[f"delta_{metric}_sem"] = sem(deltas)
        summaries.append(summary)
    summaries.sort(key=lambda row: -row["delta_p_probe_mean"])
    return summaries


def to_markdown(summaries: list[dict], target_attribute: str, miss_rate: float) -> str:
    header = (
        "| variant | n | P_probe | dP_probe | P_clip | dP_clip | id_sim | d_id_sim "
        "| CLIP-T | dCLIP-T |"
    )
    lines = [
        f"# Face steering scores (target: {target_attribute}; face miss rate {miss_rate:.1%})",
        "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['p_probe_n']}"
            f" | {row['p_probe_mean']:.3f}±{row['p_probe_sem']:.3f}"
            f" | {row['delta_p_probe_mean']:+.3f}±{row['delta_p_probe_sem']:.3f}"
            f" | {row['p_target_mean']:.3f}±{row['p_target_sem']:.3f}"
            f" | {row['delta_p_target_mean']:+.3f}±{row['delta_p_target_sem']:.3f}"
            f" | {row['id_sim_mean']:.3f}±{row['id_sim_sem']:.3f}"
            f" | {row['delta_id_sim_mean']:+.3f}±{row['delta_id_sim_sem']:.3f}"
            f" | {row['clip_t_mean']:.3f}±{row['clip_t_sem']:.3f}"
            f" | {row['delta_clip_t_mean']:+.3f}±{row['delta_clip_t_sem']:.3f} |"
        )
    return "\n".join(lines) + "\n"


@torch.no_grad()
def main(
    run_dir: str,
    target_attribute: str,
    probe_path: str,
    clip_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    """Score every image in a face steering generate manifest; write JSON + markdown.

    Args:
        run_dir: A steering generate output directory containing manifest.json.
        target_attribute: CelebA attribute name (key of ATTRIBUTE_PROMPTS).
        probe_path: Trained CelebA-HQ attribute probe bundle (.pt).
        clip_model: HF id of the CLIP checkpoint for both CLIP metrics.
        device: Torch device.
        batch_size: Images per model forward.
    """
    load_dotenv()
    if target_attribute not in ATTRIBUTE_PROMPTS:
        raise ValueError(f"No CLIP prompts for {target_attribute!r}")
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [dict(row) for row in manifest["images"]]
    images = [Image.open(run / row["path"]).convert("RGB") for row in rows]

    processor, model = get_clip(clip_model, device)
    model.eval()

    def text_features(texts: list[str]) -> torch.Tensor:
        tokens = processor(text=texts, return_tensors="pt", padding=True).to(device)
        features = output_to_feature_tensor(model.get_text_features(**tokens))
        return torch.nn.functional.normalize(features.float().cpu(), dim=-1)

    attribute_features = text_features(list(ATTRIBUTE_PROMPTS[target_attribute]))
    prompts = sorted({row["prompt"] for row in rows})
    prompt_features = {p: f for p, f in zip(prompts, text_features(prompts))}
    image_features = get_clip_features(images, processor, model, batch_size, device)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    assert image_features.shape == (len(rows), attribute_features.shape[1])
    attribute_probs = (100.0 * image_features @ attribute_features.T).softmax(dim=-1)
    for row, probs, feature in zip(rows, attribute_probs, image_features):
        row["p_target"] = probs[0].item()
        row["clip_t"] = (feature @ prompt_features[row["prompt"]]).item()

    detector = MTCNN(device=device)
    identity_model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    boxes = [face_box(detector, image) for image in images]
    found = [box is not None for box in boxes]
    miss_rate = 1 - sum(found) / len(found)
    print(f"faces found: {sum(found)}/{len(found)} (miss rate {miss_rate:.1%})", flush=True)

    bundle, probe_processor, probe_model = load_probe(probe_path, device)
    target_index = bundle["labels"].index(target_attribute)
    crops = [probe_crop(img, box) for img, box in zip(images, boxes) if box is not None]
    probs = face_probe_probs(crops, bundle, probe_processor, probe_model, batch_size, device)
    probs_iter = iter(probs)
    for row, box in zip(rows, boxes):
        if box is None:
            row["p_probe"] = float("nan")
            row["probe_attrs"] = None
        else:
            attr_probs = next(probs_iter)
            row["p_probe"] = attr_probs[target_index].item()
            row["probe_attrs"] = {
                name: attr_probs[i].item()
                for i, name in enumerate(bundle["labels"])
                if name in ("Male", "Young", "Smiling", "Eyeglasses")
            }

    embeddings: list[torch.Tensor | None] = []
    for image, box in zip(images, boxes):
        if box is None:
            embeddings.append(None)
            continue
        face = extract_face(image, box, image_size=160).to(device)
        face = (face - 127.5) / 128.0
        embedding = identity_model(face.unsqueeze(0))[0].cpu()
        embeddings.append(torch.nn.functional.normalize(embedding, dim=-1))
    base_embedding = {
        (row["prompt"], row["seed"]): emb
        for row, emb in zip(rows, embeddings)
        if row["method"] == "base"
    }
    for row, emb in zip(rows, embeddings):
        if row["method"] == "base":
            row["id_sim"] = 1.0 if emb is not None else float("nan")
            continue
        base_emb = base_embedding.get((row["prompt"], row["seed"]))
        if emb is None or base_emb is None:
            row["id_sim"] = float("nan")
        else:
            row["id_sim"] = (emb @ base_emb).item()

    summaries = aggregate_faces(rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = run.parent / f"scores_faces_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "run_dir": str(run),
        "target_attribute": target_attribute,
        "clip_model": clip_model,
        "attribute_prompts": ATTRIBUTE_PROMPTS[target_attribute],
        "probe": {
            "probe_path": str(probe_path),
            "backbone": bundle["backbone"],
            "val_metrics": bundle["val_metrics_per_attribute"][target_attribute],
        },
        "identity_model": "facenet-pytorch InceptionResnetV1 vggface2",
        "face_miss_rate": miss_rate,
        "git_sha": commit,
        "timestamp": timestamp,
        "summaries": summaries,
        "images": rows,
    }
    (out / f"{timestamp}_face_scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = to_markdown(summaries, target_attribute, miss_rate)
    (out / f"{timestamp}_face_scores.md").write_text(markdown)
    print(markdown)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
