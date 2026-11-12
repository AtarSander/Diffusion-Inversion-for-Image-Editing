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


def render(
    path: Path,
    center: np.ndarray,
    radius: float,
    angle: float,
    color,
    inner_radius: float | None = None,
) -> None:
    """Render one triangle; with inner_radius, a concentric white cutout makes a ring."""
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.set_axis_off()
    ax.add_patch(Polygon(vertices(center, radius, angle), closed=True, facecolor=color))
    if inner_radius is not None:
        ax.add_patch(
            Polygon(vertices(center, inner_radius, angle), closed=True, facecolor="white")
        )
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def ink_pixels(path: Path, threshold: float = 40.0) -> int:
    """Count pixels differing from white by more than threshold in any channel."""
    rgb = np.asarray(plt.imread(path))[..., :3] * 255.0
    return int((np.abs(rgb - 255.0).max(axis=-1) > threshold).sum())


def main(
    out_dir: str = "data/steering/triangles64_size",
    pairs: int = 64,
    big_radius: float = 300.0,
    small_radius: float = 100.0,
    seed: int = 0,
    outline: bool = False,
    ink_area: float = 10000.0,
) -> None:
    """Write `pairs` big/small matched triangle images plus a sources.jsonl manifest.

    Pair k differs ONLY in size: color, rotation, and center are shared, and the
    center is sampled so the big triangle always fits. Manifest rows interleave
    positive (big) / negative (small) so the kth positive pairs with the kth negative.

    With `outline`, each triangle is a ring (concentric white cutout) whose colored
    area equals `ink_area` px^2 for BOTH sizes, so big vs small differ in extent but
    not in ink mass. RNG draws are identical to the filled dataset at equal seed.
    """
    area_coeff = 3.0 * math.sqrt(3.0) / 4.0  # filled area = coeff * circumradius^2
    inner = {}
    if outline:
        assert ink_area < area_coeff * small_radius**2, "ink_area exceeds small triangle"
        inner = {r: math.sqrt(r**2 - ink_area / area_coeff) for r in (big_radius, small_radius)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    suffix = "outline" if outline else "size"
    ink = {"big": [], "small": []}
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
            render(out / f"{stem}.png", center, radius, angle, color, inner.get(radius))
            rows.append({"id": f"{label}_{k:03d}_{suffix}", "path": f"{stem}.png", "label": label})
            if outline:
                ink[stem.split("_")[0]].append(ink_pixels(out / f"{stem}.png"))
        print(f"pair {k + 1}/{pairs}", flush=True)
    if outline:
        mean_big = float(np.mean(ink["big"]))
        mean_small = float(np.mean(ink["small"]))
        print(f"measured ink px: big {mean_big:.0f}, small {mean_small:.0f}")
        assert abs(math.log(mean_big / mean_small)) < 0.07, (mean_big, mean_small)
    (out / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "pairs": pairs,
        "big_radius": big_radius,
        "small_radius": small_radius,
        "seed": seed,
        "size": 1024,
        "pairing": "same color/rotation/center per pair; only circumradius differs",
    }
    if outline:
        meta["outline"] = {
            "ink_area": ink_area,
            "inner_radius_big": inner[big_radius],
            "inner_radius_small": inner[small_radius],
            "measured_ink_big": mean_big,
            "measured_ink_small": mean_small,
        }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {2 * pairs} images + sources.jsonl to {out}")


if __name__ == "__main__":
    fire.Fire(main)
