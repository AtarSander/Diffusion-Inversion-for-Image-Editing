# ABOUTME: Renders the synthetic triangle-size steering dataset: matched pairs of one
# ABOUTME: equilateral triangle (big=positive vs small=negative), same color/rotation/center.
import json
import math
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def vertices(center: np.ndarray, radius: float, angle: float) -> np.ndarray:
    """Equilateral triangle vertices (circumradius param) rotated by angle."""
    angles = angle + math.pi / 2 + np.arange(3) * (2 * math.pi / 3)
    return center[None, :] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def render(path: Path, center: np.ndarray, radius: float, angle: float, color) -> None:
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.set_axis_off()
    ax.add_patch(Polygon(vertices(center, radius, angle), closed=True, facecolor=color))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main(
    out_dir: str = "data/steering/triangles64_size",
    pairs: int = 64,
    big_radius: float = 300.0,
    small_radius: float = 100.0,
    seed: int = 0,
) -> None:
    """Write `pairs` big/small matched triangle images plus a sources.jsonl manifest.

    Pair k differs ONLY in size: color, rotation, and center are shared, and the
    center is sampled so the big triangle always fits. Manifest rows interleave
    positive (big) / negative (small) so the kth positive pairs with the kth negative.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(pairs):
        margin = big_radius + 24
        center = rng.uniform(margin, 1024 - margin, size=2)
        angle = rng.uniform(0, 2 * math.pi)
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.65, 1.0)
        val = rng.uniform(0.55, 0.95)
        color = matplotlib.colors.hsv_to_rgb([hue, sat, val])
        for label, radius, stem in (
            ("positive", big_radius, f"big_{k:03d}"),
            ("negative", small_radius, f"small_{k:03d}"),
        ):
            render(out / f"{stem}.png", center, radius, angle, color)
            rows.append({"id": f"{label}_{k:03d}_size", "path": f"{stem}.png", "label": label})
        print(f"pair {k + 1}/{pairs}", flush=True)
    (out / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "pairs": pairs,
        "big_radius": big_radius,
        "small_radius": small_radius,
        "seed": seed,
        "size": 1024,
        "pairing": "same color/rotation/center per pair; only circumradius differs",
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {2 * pairs} images + sources.jsonl to {out}")


if __name__ == "__main__":
    fire.Fire(main)
