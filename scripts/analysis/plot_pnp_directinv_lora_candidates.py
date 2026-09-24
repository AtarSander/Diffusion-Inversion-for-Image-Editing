#!/usr/bin/env python3
"""Render PnP candidates with complete DirectInv and LoRA+DirectInv metrics."""
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .plot_directinv_lora_showcase import (
        BASELINE_COLOR, BOTTOM_MARGIN, COLUMN_GAP, FONT_BOLD, FONT_REGULAR,
        METRICS, OUR_COLOR, PANEL_SIZE, RIGHT_MARGIN, ROW_GAP, TOP_MARGIN,
        edited_tile, font, format_metric, metric_row, metric_value, source_image,
    )
except ImportError:
    from plot_directinv_lora_showcase import (
        BASELINE_COLOR, BOTTOM_MARGIN, COLUMN_GAP, FONT_BOLD, FONT_REGULAR,
        METRICS, OUR_COLOR, PANEL_SIZE, RIGHT_MARGIN, ROW_GAP, TOP_MARGIN,
        edited_tile, font, format_metric, metric_row, metric_value, source_image,
    )


@dataclass(frozen=True)
class Candidate:
    label: str
    file_id: str
    edit_label: str


CANDIDATES = (
    Candidate("A", "212000000001", "Add a facial tattoo"),
    Candidate("B", "000000000110", "Make the woman older"),
    Candidate("C", "211000000001", "Add a gold chain and star"),
    Candidate("D", "622000000001", "Change the dress to red"),
    Candidate("E", "613000000001", "Red → yellow curtain"),
    Candidate("F", "214000000001", "Add deer to the forest"),
    Candidate("G", "114000000007", "Spaceship → eagle"),
    Candidate("H", "000000000102", "Change the rail to wood"),
    Candidate("I", "000000000062", "Rabbit → cat"),
    Candidate("J", "222000000004", "Add a tennis racket"),
    Candidate("K", "224000000009", "Add a grazing goat"),
    Candidate("L", "823000000008", "Wall → forest"),
)

LEFT_MARGIN = 560
ROWS_PER_PAGE = 4
RESULT_DIR = "one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors"


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    selected_font,
    *,
    width: int,
) -> int:
    wrapped = textwrap.fill(value, width=width)
    draw.multiline_text(xy, wrapped, font=selected_font, fill="black", spacing=5)
    return draw.multiline_textbbox(xy, wrapped, font=selected_font, spacing=5)[3]


def render_page(
    rows: tuple[Candidate, ...],
    mapping: dict,
    data_root: Path,
    evaluation_root: Path,
    editor_label: str,
    editor_dir: str,
    result_dir: str,
) -> Image.Image:
    width = LEFT_MARGIN + 3 * PANEL_SIZE + 2 * COLUMN_GAP + RIGHT_MARGIN
    height = (
        TOP_MARGIN
        + len(rows) * PANEL_SIZE
        + (len(rows) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = font(FONT_BOLD, 30)
    label_font = font(FONT_BOLD, 34)
    editor_font = font(FONT_BOLD, 27)
    edit_font = font(FONT_REGULAR, 25)
    metric_header_font = font(FONT_BOLD, 21)
    metric_font = font(FONT_REGULAR, 22)
    metric_name_font = font(FONT_BOLD, 22)

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

    baseline_csv = (
        evaluation_root
        / "sd15_fp16_baselines"
        / "metrics"
        / f"directinversion+{editor_dir}.csv"
    )
    ours_csv = (
        evaluation_root
        / result_dir
        / "metrics"
        / f"lora+directinversion+{editor_dir}.csv"
    )

    for row_index, row in enumerate(rows):
        relative_path = Path(mapping[row.file_id]["image_path"])
        y = TOP_MARGIN + row_index * (PANEL_SIZE + ROW_GAP)
        source = source_image(data_root / "annotation_images" / relative_path)
        direct = edited_tile(
            evaluation_root
            / "sd15_fp16_baselines"
            / f"directinversion+{editor_dir}"
            / "annotation_images"
            / relative_path
        )
        ours = edited_tile(
            evaluation_root
            / result_dir
            / f"lora+directinversion+{editor_dir}"
            / "annotation_images"
            / relative_path
        )
        baseline_metrics = metric_row(baseline_csv, row.file_id)
        ours_metrics = metric_row(ours_csv, row.file_id)

        for column, image in enumerate((source, direct, ours)):
            x = LEFT_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + PANEL_SIZE - 1, y + PANEL_SIZE - 1),
                outline=OUR_COLOR if column == 2 else "#B5B5B5",
                width=5 if column == 2 else 2,
            )

        text_y = y + 6
        draw.text((20, text_y), row.label, font=label_font, fill=OUR_COLOR)
        draw.text((76, text_y + 4), editor_label, font=editor_font, fill="black")
        text_y += 55
        text_y = draw_wrapped(
            draw, (20, text_y), row.edit_label, edit_font, width=27
        ) + 18

        metric_x = 20
        baseline_x = 270
        arrow_x = 363
        ours_x = 405
        draw.text((metric_x, text_y), "Metric", font=metric_header_font, fill="black")
        draw.text(
            (baseline_x, text_y),
            "DirectInv",
            font=metric_header_font,
            fill=BASELINE_COLOR,
        )
        draw.text((ours_x, text_y), "Ours", font=metric_header_font, fill=OUR_COLOR)
        text_y += 39

        for metric_name, suffix, precision in METRICS:
            baseline_value = format_metric(
                metric_value(baseline_metrics, suffix), precision
            )
            ours_value = format_metric(metric_value(ours_metrics, suffix), precision)
            draw.text((metric_x, text_y), metric_name, font=metric_name_font, fill="black")
            draw.text(
                (baseline_x, text_y),
                baseline_value,
                font=metric_font,
                fill=BASELINE_COLOR,
            )
            draw.text((arrow_x, text_y), "→", font=metric_font, fill="#777777")
            draw.text((ours_x, text_y), ours_value, font=metric_font, fill=OUR_COLOR)
            text_y += 38

    return canvas


def render(
    data_root: Path,
    evaluation_root: Path,
    output: Path,
    *,
    candidates: tuple[Candidate, ...] = CANDIDATES,
    editor_label: str = "PnP",
    editor_dir: str = "pnp",
    result_dir: str = RESULT_DIR,
) -> None:
    mapping = json.loads((data_root / "mapping_file.json").read_text())
    pages = [
        render_page(
            candidates[start : start + ROWS_PER_PAGE],
            mapping,
            data_root,
            evaluation_root,
            editor_label,
            editor_dir,
            result_dir,
        )
        for start in range(0, len(candidates), ROWS_PER_PAGE)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for page_number, page in enumerate(pages, start=1):
        page.save(
            output.with_name(f"{output.stem}_page{page_number}.png"),
            format="PNG",
            dpi=(300, 300),
            optimize=True,
        )
    pages[0].save(
        output.with_suffix(".pdf"),
        format="PDF",
        resolution=300.0,
        save_all=True,
        append_images=pages[1:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/PIE-Bench_v1")
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/pnp_directinv_lora_candidates.png"),
    )
    args = parser.parse_args()
    render(args.data_root, args.evaluation_root, args.output)


if __name__ == "__main__":
    main()
