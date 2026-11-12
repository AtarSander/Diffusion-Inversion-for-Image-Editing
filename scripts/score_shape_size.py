# ABOUTME: Scores an edit run for shape size: colored-area equivalent radius per
# ABOUTME: image, aggregated per (method, strength) as the ratio to the input image.
import json
import math
import statistics
import time
from pathlib import Path

import fire
import numpy as np
from PIL import Image


def eq_radius(path: str | Path, threshold: float = 40.0) -> float:
    """Equivalent radius sqrt(area/pi) of pixels differing from white by > threshold."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    area = (np.abs(rgb - 255.0).max(axis=-1) > threshold).sum()
    return math.sqrt(area / math.pi)


def sem(values: list[float]) -> float:
    """Standard error of the mean (0 for n<=1)."""
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def main(run_dir: str, threshold: float = 40.0) -> None:
    """Score one edit run: per-image radius ratios and a per-(method, strength) table."""
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    inputs = {row["source_id"]: row["source_path"] for row in manifest["images"]}
    input_radius = {sid: eq_radius(path, threshold) for sid, path in inputs.items()}
    rows = []
    for row in manifest["images"]:
        radius = eq_radius(run / row["path"], threshold)
        rows.append(
            {
                "source_id": row["source_id"],
                "method": row["method"],
                "strength": row["strength"],
                "eq_radius": radius,
                "ratio_vs_input": radius / input_radius[row["source_id"]],
            }
        )
    ts = time.strftime("%Y%m%d_%H%M%S")
    lines = [
        f"# Shape-size scores ({ts})",
        "",
        f"Run: {run}. Equivalent radius sqrt(area/pi) of non-white pixels "
        f"(threshold {threshold}); ratio is vs each source image.",
        "",
        "| method | strength | ratio_vs_input | n |",
        "|---|---|---|---|",
    ]
    for method in ("DDIM", "MW"):
        strengths = sorted({r["strength"] for r in rows if r["method"] == method})
        for strength in strengths:
            ratios = [
                r["ratio_vs_input"]
                for r in rows
                if r["method"] == method and r["strength"] == strength
            ]
            lines.append(
                f"| {method} | {strength:+.1f} | "
                f"{statistics.mean(ratios):.3f}±{sem(ratios):.3f} | {len(ratios)} |"
            )
    out_json = run / f"scores_size_{ts}.json"
    out_json.write_text(
        json.dumps({"threshold": threshold, "input_radius": input_radius, "rows": rows}, indent=2)
        + "\n"
    )
    out_md = run / f"scores_size_{ts}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    fire.Fire(main)
