#!/usr/bin/env python3
"""Create labeled 5x5 PIE-Bench reconstruction contact sheets from selections."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

EDITORS = ("pix2pix-zero", "masactrl", "pnp")
LABEL_HEIGHT = 28
GRID = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for editor in EDITORS:
        for kind in ("best", "worst"):
            source_dir = args.selection_root / editor / kind
            with (args.selection_root / editor / f"{kind}.csv").open(newline="", encoding="utf-8") as handle:
                entries = list(csv.DictReader(handle))
            if len(entries) != GRID * GRID:
                raise ValueError(f"Expected 25 {editor}/{kind} entries, got {len(entries)}")
            panels = [Image.open(source_dir / entry["panel"]).convert("RGB") for entry in entries]
            panel_w = max(image.width for image in panels)
            panel_h = max(image.height for image in panels)
            canvas = Image.new("RGB", (GRID * panel_w, GRID * (panel_h + LABEL_HEIGHT)), "white")
            draw = ImageDraw.Draw(canvas)
            for index, (image, entry) in enumerate(zip(panels, entries)):
                row, col = divmod(index, GRID)
                x, y = col * panel_w, row * (panel_h + LABEL_HEIGHT)
                canvas.paste(image, (x, y))
                score = float(entry["composite_reconstruction_rank"])
                draw.text((x + 3, y + panel_h + 7), f"#{entry['rank']}  {entry['file_id']}  {score:.3f}", fill="black", font=font)
            filename = f"{editor}_{kind}_5x5.jpg"
            canvas.save(args.output / filename, quality=95, subsampling=0)
            print(args.output / filename)

if __name__ == "__main__":
    main()
