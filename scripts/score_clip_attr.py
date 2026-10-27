# ABOUTME: Back-fills clip_t_attr (CLIP similarity to the positive attribute prompt)
# ABOUTME: for existing face-steering runs, one sidecar JSON per run directory.
import glob
import json
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
from diff_inversion.steering.score_faces import ATTRIBUTE_PROMPTS

RUN_GLOBS = {
    "Smiling": ["output/steering/celeba_smiling/*", "output/steering/celeba_layers/smiling_*/*"],
    "Eyeglasses": [
        "output/steering/celeba_eyeglasses/*",
        "output/steering/celeba_layers/eyeglasses_*/*",
    ],
    "Young": ["output/steering/celeba_young/*", "output/steering/celeba_layers/young_*/*"],
}


@torch.no_grad()
def score_run(run_base: Path, attribute: str, processor, model, device, batch_size) -> None:
    """Write <run_base>/clip_attr_scores_<ts>.json for the run's generate manifest."""
    manifest = json.loads((run_base / "generate" / "manifest.json").read_text())
    rows = [dict(row) for row in manifest["images"]]
    images = [
        Image.open(run_base / "generate" / row["path"]).convert("RGB") for row in rows
    ]
    tokens = processor(
        text=[ATTRIBUTE_PROMPTS[attribute][0]], return_tensors="pt", padding=True
    ).to(device)
    text = output_to_feature_tensor(model.get_text_features(**tokens))
    text = torch.nn.functional.normalize(text.float().cpu(), dim=-1)[0]
    features = get_clip_features(images, processor, model, batch_size, device)
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    assert features.shape == (len(rows), text.shape[0])
    out_rows = [
        {
            "prompt": row["prompt"],
            "seed": row["seed"],
            "method": row["method"],
            "strength": row["strength"],
            "path": row["path"],
            "clip_t_attr": (feature @ text).item(),
        }
        for row, feature in zip(rows, features)
    ]
    out = run_base / f"clip_attr_scores_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "attribute": attribute,
                "attribute_prompt": ATTRIBUTE_PROMPTS[attribute][0],
                "rows": out_rows,
            }
        )
        + "\n"
    )
    print(f"wrote {out} ({len(out_rows)} rows)")


def main(
    clip_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    """Score every discovered CelebA steering run that has a generate manifest."""
    load_dotenv()
    processor, model = get_clip(clip_model, device)
    model.eval()
    for attribute, patterns in RUN_GLOBS.items():
        for pattern in patterns:
            for base in sorted(glob.glob(pattern)):
                base = Path(base)
                if not (base / "generate" / "manifest.json").is_file():
                    continue
                score_run(base, attribute, processor, model, device, batch_size)


if __name__ == "__main__":
    fire.Fire(main)
