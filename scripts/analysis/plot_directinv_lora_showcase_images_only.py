#!/usr/bin/env python3
"""Render the selected qualitative comparison without metric annotations."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from plot_directinv_lora_showcase import (
    BOTTOM_MARGIN,
    COLUMN_GAP,
    FONT_BOLD,
    FONT_REGULAR,
    OUR_COLOR,
    PANEL_SIZE,
    RIGHT_MARGIN,
    ROWS,
    ROW_GAP,
    TOP_MARGIN,
    edited_tile,
    font,
    source_image,
)


LEFT_MARGIN = 360


def render(data_root: Path, evaluation_root: Path, output: Path) -> None:
    mapping = json.loads((data_root / "mapping_file.json").read_text())
    width = LEFT_MARGIN + 3 * PANEL_SIZE + 2 * COLUMN_GAP + RIGHT_MARGIN
    height = (
        TOP_MARGIN
        + len(ROWS) * PANEL_SIZE
        + (len(ROWS) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(FONT_BOLD, 38)
    editor_font = font(FONT_BOLD, 44)
    edit_font = font(FONT_REGULAR, 30)

    for column, title in enumerate(("Source", "DirectInv", "LoRA + DirectInv (Ours)")):
        x = LEFT_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
        color = OUR_COLOR if column == 2 else "black"
        box = draw.textbbox((0, 0), title, font=header_font)
        draw.text(
            (x + (PANEL_SIZE - (box[2] - box[0])) / 2, 30),
            title,
            font=header_font,
            fill=color,
        )

    for row_index, row in enumerate(ROWS):
        relative_path = Path(mapping[row.file_id]["image_path"])
        y = TOP_MARGIN + row_index * (PANEL_SIZE + ROW_GAP)
        source = source_image(data_root / "annotation_images" / relative_path)
        direct = edited_tile(
            evaluation_root
            / "sd15_fp16_baselines"
            / f"directinversion+{row.editor_dir}"
            / "annotation_images"
            / relative_path
        )
        ours = edited_tile(
            evaluation_root
            / row.result_dir
            / f"lora+directinversion+{row.editor_dir}"
            / "annotation_images"
            / relative_path
        )

        wrapped_edit = textwrap.fill(row.edit_label, width=20)
        editor_box = draw.textbbox((0, 0), row.editor, font=editor_font)
        edit_box = draw.multiline_textbbox(
            (0, 0), wrapped_edit, font=edit_font, spacing=6
        )
        block_height = (
            editor_box[3] - editor_box[1]
            + 18
            + edit_box[3] - edit_box[1]
        )
        label_y = y + (PANEL_SIZE - block_height) / 2
        draw.text(
            (20, label_y),
            row.editor,
            font=editor_font,
            fill="black",
        )
        draw.multiline_text(
            (20, label_y + editor_box[3] - editor_box[1] + 18),
            wrapped_edit,
            font=edit_font,
            fill="black",
            spacing=6,
        )

        for column, image in enumerate((source, direct, ours)):
            x = LEFT_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + PANEL_SIZE - 1, y + PANEL_SIZE - 1),
                outline=OUR_COLOR if column == 2 else "#B5B5B5",
                width=5 if column == 2 else 2,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(300, 300), optimize=True)
    canvas.save(output.with_suffix(".pdf"), format="PDF", resolution=300.0)
    print(output)
    print(output.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/PIE-Bench_v1")
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/figures/directinv_lora_qualitative_comparison_images_only.png"
        ),
    )
    args = parser.parse_args()
    render(args.data_root, args.evaluation_root, args.output)


if __name__ == "__main__":
    main()
