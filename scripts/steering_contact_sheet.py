# ABOUTME: Builds a self-contained HTML contact sheet for steering generate runs:
# ABOUTME: one row per prompt, base vs selected variants, captioned with CLIP scores.
"""Render steering generations into one HTML file with inline thumbnails.

Usage:
    uv run python scripts/steering_contact_sheet.py \
        --runs '{"Ukiyo_e": "output/steering/ukiyo_e/<ts>", "Fauvism": "..."}' \
        --out output/steering/<ts>_contact_sheet.html
"""

import base64
import io
import json
import time
from pathlib import Path

import fire
from PIL import Image


def thumbnail(path: Path, width: int = 320) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def latest_scores(run_dir: Path, generate_name: str) -> dict[str, dict]:
    """Map image path -> scored row from the newest scores JSON for a generate dir."""
    candidates = sorted(run_dir.glob("scores_*/*_clip_scores.json"))
    for path in reversed(candidates):
        payload = json.loads(path.read_text())
        if Path(payload["run_dir"]).name == generate_name:
            return {row["path"]: row for row in payload["images"]}
    return {}


def main(
    runs: dict[str, str],
    out: str,
    generate_name: str = "generate",
    variants: tuple[str, ...] = (
        "SA_DDIM_true_2.0",
        "SA_MW_true_2.0",
        "SA_DDIM_true_4.0",
        "SA_MW_true_4.0",
        "SA_DDIM_true_8.0",
        "SA_MW_true_8.0",
    ),
) -> None:
    """Write the contact sheet HTML.

    Args:
        runs: Mapping of section title (target style) to a steering run directory
            that contains `<generate_name>/manifest.json`.
        out: Output HTML path.
        generate_name: Generate subdirectory to render (e.g. "generate_pass2").
        variants: Image stems shown after the base column, in order.
    """
    sections = []
    for title, run in runs.items():
        run_dir = Path(run)
        generate = run_dir / generate_name
        manifest = json.loads((generate / "manifest.json").read_text())
        scores = latest_scores(run_dir, generate_name)
        by_unit: dict[tuple[str, int], dict[str, str]] = {}
        for row in manifest["images"]:
            stem = Path(row["path"]).stem
            by_unit.setdefault((row["prompt"], row["seed"]), {})[stem] = row["path"]
        header = "".join(
            f"<th>{name}</th>" for name in ("prompt / seed", "base", *variants)
        )
        body_rows = []
        for (prompt, seed), stems in sorted(by_unit.items()):
            cells = [f"<td class='prompt'>{prompt}<br><small>seed {seed}</small></td>"]
            for stem in ("base", *variants):
                if stem not in stems:
                    cells.append("<td>—</td>")
                    continue
                path = stems[stem]
                caption = ""
                if path in scores:
                    row = scores[path]
                    caption = (
                        f"P={row['p_target']:.3f} · CLIP-T={row['clip_t']:.3f}"
                    )
                cells.append(
                    f"<td><img src='data:image/jpeg;base64,{thumbnail(generate / path)}'>"
                    f"<div class='cap'>{caption}</div></td>"
                )
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        sections.append(
            f"<h2>{title} <small>({generate})</small></h2>"
            f"<table><tr>{header}</tr>{''.join(body_rows)}</table>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Steering contact sheet</title><style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 16px; }}
table {{ border-collapse: collapse; margin-bottom: 32px; }}
td, th {{ border: 1px solid #333; padding: 4px; text-align: center; vertical-align: top; }}
td.prompt {{ max-width: 130px; font-size: 12px; text-align: left; }}
img {{ width: 320px; display: block; }}
.cap {{ font-size: 11px; color: #9cf; padding-top: 2px; }}
h2 small {{ color: #888; font-size: 60%; }}
</style></head><body>
<h1>Image-derived steering — {time.strftime("%Y-%m-%d %H:%M")}</h1>
<p>P = CLIP zero-shot probability of the target style (27 WikiArt styles);
CLIP-T = adherence to the generation prompt.</p>
{"".join(sections)}</body></html>"""
    Path(out).write_text(html)
    print(f"Wrote {out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    fire.Fire(main)
