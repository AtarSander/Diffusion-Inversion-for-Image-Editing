#!/usr/bin/env python3
"""Render the selected DirectInv versus LoRA+DirectInv qualitative comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Row:
    editor: str
    editor_dir: str
    file_id: str
    edit_label: str
    result_dir: str


ROWS = (
    Row(
        editor="P2P",
        editor_dir="p2p",
        file_id="111000000005",
        edit_label="Dog → monkey",
        result_dir="one_epoch/cycle_loss_cfg1_r16_lambda0p5_uniform_e1_all_editors",
    ),
    Row(
        editor="PnP",
        editor_dir="pnp",
        file_id="212000000001",
        edit_label="Add a facial tattoo",
        result_dir="one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors",
    ),
    Row(
        editor="MasaCtrl",
        editor_dir="masactrl",
        file_id="311000000002",
        edit_label="Remove the laptop",
        result_dir="one_epoch/cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors",
    ),
    Row(
        editor="Pix2Pix-Zero",
        editor_dir="pix2pix-zero",
        file_id="000000000001",
        edit_label="Round → square cake",
        result_dir="one_epoch/cycle_loss_cfg1_r32_uniform_e1_all_editors",
    ),
)

METRICS = (
    ("Structure ↓", "structure_distance", 4),
    ("PSNR ↑", "psnr_unedit_part", 2),
    ("LPIPS ↓", "lpips_unedit_part", 3),
    ("MSE ↓", "mse_unedit_part", 4),
    ("SSIM ↑", "ssim_unedit_part", 3),
    ("CLIP target ↑", "clip_similarity_target_image", 2),
    ("CLIP edit ↑", "clip_similarity_target_image_edit_part", 2),
)

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
PANEL_SIZE = 512
LEFT_MARGIN = 560
RIGHT_MARGIN = 28
TOP_MARGIN = 92
BOTTOM_MARGIN = 28
COLUMN_GAP = 18
ROW_GAP = 92
OUR_COLOR = "#0072B2"
BASELINE_COLOR = "#4A4A4A"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def edited_tile(path: Path) -> Image.Image:
    with Image.open(path) as panel:
        panel = panel.convert("RGB")
        if panel.width == panel.height:
            tile = panel.copy()
        else:
            tile = panel.crop((panel.width - panel.height, 0, panel.width, panel.height))
    return tile.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)


def source_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(
            (PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS
        )


def metric_row(path: Path, file_id: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["file_id"] == file_id:
                return row
    raise KeyError(f"{file_id} is missing from {path}")


def metric_value(row: dict[str, str], suffix: str) -> float:
    key = next(key for key in row if key.endswith(f"|{suffix}"))
    return float(row[key])


def format_metric(value: float, precision: int) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.{precision}f}"


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    selected_font: ImageFont.FreeTypeFont,
    *,
    fill: str,
    width: int,
    spacing: int = 5,
) -> int:
    wrapped = textwrap.fill(value, width=width)
    draw.multiline_text(
        position,
        wrapped,
        font=selected_font,
        fill=fill,
        spacing=spacing,
    )
    box = draw.multiline_textbbox(position, wrapped, font=selected_font, spacing=spacing)
    return box[3]


def render(data_root: Path, evaluation_root: Path, output: Path) -> None:
    mapping = json.loads((data_root / "mapping_file.json").read_text())
    columns = ("Source", "DirectInv", "LoRA + DirectInv (Ours)")
    width = LEFT_MARGIN + 3 * PANEL_SIZE + 2 * COLUMN_GAP + RIGHT_MARGIN
    height = TOP_MARGIN + len(ROWS) * PANEL_SIZE + (len(ROWS) - 1) * ROW_GAP + BOTTOM_MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = font(FONT_BOLD, 30)
    editor_font = font(FONT_BOLD, 30)
    edit_font = font(FONT_REGULAR, 24)
    metric_header_font = font(FONT_BOLD, 21)
    metric_font = font(FONT_REGULAR, 22)
    metric_name_font = font(FONT_BOLD, 22)

    for column, title in enumerate(columns):
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
        item = mapping[row.file_id]
        relative_path = Path(item["image_path"])
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

        baseline_metrics = metric_row(
            evaluation_root
            / "sd15_fp16_baselines"
            / "metrics"
            / f"directinversion+{row.editor_dir}.csv",
            row.file_id,
        )
        ours_metrics = metric_row(
            evaluation_root
            / row.result_dir
            / "metrics"
            / f"lora+directinversion+{row.editor_dir}.csv",
            row.file_id,
        )

        for column, image in enumerate((source, direct, ours)):
            x = LEFT_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
            canvas.paste(image, (x, y))
            outline = OUR_COLOR if column == 2 else "#B5B5B5"
            line_width = 5 if column == 2 else 2
            draw.rectangle(
                (x, y, x + PANEL_SIZE - 1, y + PANEL_SIZE - 1),
                outline=outline,
                width=line_width,
            )

        text_y = y + 8
        draw.text((20, text_y), row.editor, font=editor_font, fill="black")
        text_y += 44
        text_y = draw_multiline(
            draw,
            (20, text_y),
            row.edit_label,
            edit_font,
            fill="black",
            width=25,
        ) + 21

        metric_x = 20
        baseline_x = 270
        arrow_x = 363
        ours_x = 405
        draw.text((metric_x, text_y), "Metric", font=metric_header_font, fill="black")
        draw.text(
            (baseline_x, text_y), "DirectInv", font=metric_header_font,
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
                (baseline_x, text_y), baseline_value,
                font=metric_font, fill=BASELINE_COLOR,
            )
            draw.text((arrow_x, text_y), "→", font=metric_font, fill="#777777")
            draw.text((ours_x, text_y), ours_value, font=metric_font, fill=OUR_COLOR)
            text_y += 38

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(300, 300), optimize=True)
    canvas.save(output.with_suffix(".pdf"), format="PDF", resolution=300.0)
    print(output)
    print(output.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/directinv_lora_qualitative_comparison.png"),
    )
    args = parser.parse_args()
    render(args.data_root, args.evaluation_root, args.output)


if __name__ == "__main__":
    main()
