# ABOUTME: Scores steering generations with CLIP: zero-shot WikiArt style probability
# ABOUTME: of the target style plus CLIP-T prompt adherence, paired against each base image.
"""CLIP scoring for steering generate outputs.

Usage:
    uv run python -m diff_inversion.steering.score \
        --run_dir output/steering/ukiyo_e/<ts>/generate --target_style Ukiyo_e

Zero-shot style probability is a softmax over the 27 WikiArt style prompts, so it
only quantifies styles CLIP can name; non-promptable concepts need the planned
feature-probe/CSD metrics instead.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from PIL import Image

from diff_inversion.eval.clip_dino import (
    get_clip,
    get_clip_features,
    output_to_feature_tensor,
)

# ClassLabel names of huggan/wikiart (fixed 27-style WikiArt taxonomy).
STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]
PROMPT_NAMES = {"Ukiyo_e": "ukiyo-e", "Mannerism_Late_Renaissance": "mannerism"}


def style_prompt(style: str) -> str:
    name = PROMPT_NAMES.get(style, style.replace("_", " ").lower())
    return f"a painting in the style of {name}"


def variant_key(row: dict) -> str:
    if row["method"] == "base":
        return "base"
    return f"{row['representation']}/{row['method']}/{row['control']}/a={row['strength']}"


def sem(values: list[float]) -> float:
    return statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0


def aggregate(rows: list[dict]) -> list[dict]:
    """Pair each variant image with its base image by (prompt, seed).

    Returns one summary per variant key with mean/sem of the target-style
    probability and CLIP-T, and the paired deltas against the base image.
    """
    base = {(row["prompt"], row["seed"]): row for row in rows if row["method"] == "base"}
    if len(base) != sum(row["method"] == "base" for row in rows):
        raise ValueError("Expected exactly one base image per (prompt, seed)")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(variant_key(row), []).append(row)
    metrics = [key for key in ("p_target", "clip_t", "p_probe", "csd") if key in rows[0]]
    summaries = []
    for key, members in groups.items():
        summary: dict = {"variant": key, "n": len(members)}
        for metric in metrics:
            values = [row[metric] for row in members]
            deltas = [row[metric] - base[(row["prompt"], row["seed"])][metric] for row in members]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sem"] = sem(values)
            summary[f"delta_{metric}_mean"] = statistics.mean(deltas)
            summary[f"delta_{metric}_sem"] = sem(deltas)
        summaries.append(summary)
    summaries.sort(key=lambda row: -row["delta_p_target_mean"])
    return summaries


def to_markdown(summaries: list[dict], target_style: str) -> str:
    extras = [
        (metric, label)
        for metric, label in (("p_probe", "P_probe"), ("csd", "CSD"))
        if f"{metric}_mean" in summaries[0]
    ]
    header = "| variant | n | P(target) | dP vs base | CLIP-T | dCLIP-T |"
    rule = "|---|---|---|---|---|---|"
    for _, label in extras:
        header += f" {label} | d{label} |"
        rule += "---|---|"
    lines = [f"# CLIP zero-shot steering scores (target: {target_style})", "", header, rule]
    for row in summaries:
        line = (
            f"| {row['variant']} | {row['n']}"
            f" | {row['p_target_mean']:.3f}±{row['p_target_sem']:.3f}"
            f" | {row['delta_p_target_mean']:+.3f}±{row['delta_p_target_sem']:.3f}"
            f" | {row['clip_t_mean']:.3f}±{row['clip_t_sem']:.3f}"
            f" | {row['delta_clip_t_mean']:+.3f}±{row['delta_clip_t_sem']:.3f} |"
        )
        for metric, _ in extras:
            line += (
                f" {row[f'{metric}_mean']:.3f}±{row[f'{metric}_sem']:.3f}"
                f" | {row[f'delta_{metric}_mean']:+.3f}±{row[f'delta_{metric}_sem']:.3f} |"
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


@torch.no_grad()
def main(
    run_dir: str,
    target_style: str,
    clip_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 16,
    probe_path: str | None = None,
    csd_refs: str | None = None,
) -> None:
    """Score every image in a generate manifest and write JSON + markdown tables.

    Args:
        run_dir: A steering generate output directory containing manifest.json.
        target_style: WikiArt style name whose probability is reported.
        clip_model: HF id of the CLIP checkpoint used for both metrics.
        device: Torch device for CLIP.
        batch_size: Images per CLIP forward.
        probe_path: Optional trained WikiArt style probe (.pt); adds text-free
            P_probe columns from DINOv2 features.
        csd_refs: Optional sources.jsonl; adds a CSD column = mean style
            similarity to its positive-labeled reference images.
    """
    load_dotenv()
    if target_style not in STYLES:
        raise ValueError(f"Unknown style {target_style!r}; expected one of {STYLES}")
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [dict(row) for row in manifest["images"]]
    processor, model = get_clip(clip_model, device)
    model.eval()

    def text_features(texts: list[str]) -> torch.Tensor:
        tokens = processor(text=texts, return_tensors="pt", padding=True).to(device)
        features = output_to_feature_tensor(model.get_text_features(**tokens))
        features = features.float().cpu()
        return torch.nn.functional.normalize(features, dim=-1)

    style_features = text_features([style_prompt(style) for style in STYLES])
    target_index = STYLES.index(target_style)
    prompts = sorted({row["prompt"] for row in rows})
    prompt_features = {prompt: feature for prompt, feature in zip(prompts, text_features(prompts))}
    images = [Image.open(run / row["path"]).convert("RGB") for row in rows]
    image_features = get_clip_features(images, processor, model, batch_size, device)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    assert image_features.shape == (len(rows), style_features.shape[1])
    style_probs = (100.0 * image_features @ style_features.T).softmax(dim=-1)
    for row, probs, feature in zip(rows, style_probs, image_features):
        row["p_target"] = probs[target_index].item()
        row["clip_t"] = (feature @ prompt_features[row["prompt"]]).item()
    probe_meta = None
    if probe_path:
        from .probe import load_probe, probe_probs

        bundle, probe_processor, probe_model = load_probe(probe_path, device)
        probe_target = bundle["labels"].index(target_style)
        probs = probe_probs(images, bundle, probe_processor, probe_model, batch_size, device)
        for row, image_probs in zip(rows, probs):
            row["p_probe"] = image_probs[probe_target].item()
        probe_meta = {
            "probe_path": str(probe_path),
            "backbone": bundle["backbone"],
            "val_accuracy": bundle["val_accuracy"],
        }
    csd_meta = None
    if csd_refs:
        from .config import read_sources
        from .csd import csd_style_features, load_csd, mean_ref_similarity

        sources = read_sources(csd_refs)
        ref_paths = [s["path"] for s in sources if s["label"] == "positive"]
        neg_paths = [s["path"] for s in sources if s["label"] == "negative"]
        csd_processor, csd_model = load_csd(device)
        ref_images = [Image.open(p).convert("RGB") for p in ref_paths]
        neg_images = [Image.open(p).convert("RGB") for p in neg_paths]
        ref_feats = csd_style_features(ref_images, csd_processor, csd_model, batch_size, device)
        neg_feats = csd_style_features(neg_images, csd_processor, csd_model, batch_size, device)
        feats = csd_style_features(images, csd_processor, csd_model, batch_size, device)
        for row, value in zip(rows, mean_ref_similarity(feats, ref_feats)):
            row["csd"] = value.item()
        pairwise = ref_feats @ ref_feats.T
        off_diag = pairwise[~torch.eye(len(ref_feats), dtype=torch.bool)]
        csd_meta = {
            "csd_refs": str(csd_refs),
            "n_refs": len(ref_feats),
            "ref_self_sim": off_diag.mean().item(),
            "neg_ref_sim": mean_ref_similarity(neg_feats, ref_feats).mean().item(),
        }
        print(
            f"CSD ceiling (ref-ref) {csd_meta['ref_self_sim']:.3f}, "
            f"floor (neg-ref) {csd_meta['neg_ref_sim']:.3f}"
        )
    summaries = aggregate(rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = run.parent / f"scores_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "run_dir": str(run),
        "target_style": target_style,
        "clip_model": clip_model,
        "probe": probe_meta,
        "csd": csd_meta,
        "git_sha": commit,
        "timestamp": timestamp,
        "summaries": summaries,
        "images": rows,
    }
    (out / f"{timestamp}_clip_scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = to_markdown(summaries, target_style)
    (out / f"{timestamp}_clip_scores.md").write_text(markdown)
    print(markdown)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
