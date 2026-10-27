# ABOUTME: Renders mid-size colored circles on white (same style as the triangle
# ABOUTME: dataset) as edit inputs for size steering, with a sources.jsonl manifest.
import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def render(path: Path, center: np.ndarray, radius: float, color) -> None:
    """Draw one filled circle on a white 1024x1024 canvas."""
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.set_axis_off()
    ax.add_patch(Circle(tuple(center), radius, facecolor=color))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main(
    out_dir: str = "data/steering/circles8_eval",
    count: int = 8,
    radius: float = 175.0,
    grow_room: float = 300.0,
    seed: int = 1,
) -> None:
    """Write `count` circle images plus a sources.jsonl manifest of {id, path} rows.

    The radius sits between the triangle dataset's small (100) and big (300)
    circumradii so a size edit is visible in both directions, and centers keep
    `grow_room` margin so a grown shape does not clip at the border.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(count):
        margin = grow_room + 24
        center = rng.uniform(margin, 1024 - margin, size=2)
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.65, 1.0)
        val = rng.uniform(0.55, 0.95)
        color = matplotlib.colors.hsv_to_rgb([hue, sat, val])
        stem = f"circle_{k:03d}"
        render(out / f"{stem}.png", center, radius, color)
        rows.append({"id": stem, "path": f"{stem}.png"})
        print(f"circle {k + 1}/{count}", flush=True)
    (out / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "count": count,
        "radius": radius,
        "grow_room": grow_room,
        "seed": seed,
        "size": 1024,
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {count} images + sources.jsonl to {out}")


if __name__ == "__main__":
    fire.Fire(main)
