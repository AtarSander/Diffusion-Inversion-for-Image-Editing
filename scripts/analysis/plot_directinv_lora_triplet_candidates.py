#!/usr/bin/env python3
"""Render twelve candidate Source/DirectInv/LoRA+DirectInv triplets.

The public-facing labels intentionally omit PIE-Bench identifiers and LoRA
hyperparameters. Candidate letters provide a compact way to choose examples;
the identifiers and result directories remain in this script solely to make
the figure reproducible.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Candidate:
    label: str
    editor: str
    editor_dir: str
    file_id: str
    edit_label: str
    headline_gain: str
    result_dir: str


CANDIDATES = (
    Candidate(
        "A", "P2P", "p2p", "111000000005", "Dog -> monkey",
        "PSNR +8.41 dB | LPIPS -0.216 | CLIP edit +5.64",
        "one_epoch/cycle_loss_cfg1_r16_lambda0p5_uniform_e1_all_editors",
    ),
    Candidate(
        "B", "P2P", "p2p", "123000000006", "Notebook -> calendar",
        "CLIP edit +8.91 | CLIP target +6.22 | PSNR +2.95 dB",
        "one_epoch/cycle_loss_cfg1_r16_lambda0p5_uniform_e1_all_editors",
    ),
    Candidate(
        "C", "P2P", "p2p", "000000000089", "Red berries -> purple berries",
        "PSNR +14.67 dB | LPIPS -0.252 | CLIP edit +2.68",
        "one_epoch/cycle_loss_cfg1_r16_lambda0p5_uniform_e1_all_editors",
    ),
    Candidate(
        "D", "PnP", "pnp", "423000000000", "Plaid -> floral carpet",
        "CLIP edit +4.95 | PSNR +2.64 dB | SSIM +0.057",
        "one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors",
    ),
    Candidate(
        "E", "PnP", "pnp", "924000000009", "Stained-glass style",
        "CLIP target/edit +13.70 | stronger edit, lower structure fidelity",
        "one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors",
    ),
    Candidate(
        "F", "PnP", "pnp", "721000000001", "Robin -> silver sculpture",
        "CLIP edit +2.15 | LPIPS -0.021 | CLIP target +1.19",
        "one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors",
    ),
    Candidate(
        "G", "MasaCtrl", "masactrl", "000000000001", "Round -> square cake",
        "PSNR +2.34 dB | LPIPS -0.021 | all metrics improve",
        "one_epoch/cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors",
    ),
    Candidate(
        "H", "MasaCtrl", "masactrl", "111000000003", "Bunny -> pig",
        "Structure +0.060 | LPIPS -0.110 | CLIP edit +2.74",
        "one_epoch/cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors",
    ),
    Candidate(
        "I", "MasaCtrl", "masactrl", "211000000001", "Add gold chain and star",
        "CLIP target +3.36 | LPIPS -0.016 | all metrics improve",
        "one_epoch/cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors",
    ),
    Candidate(
        "J", "Pix2Pix-Zero", "pix2pix-zero", "124000000004",
        "Plant -> seated man",
        "CLIP edit +8.19 | LPIPS -0.210 | SSIM +0.218",
        "one_epoch/cycle_loss_cfg1_r32_uniform_e1_all_editors",
    ),
    Candidate(
        "K", "Pix2Pix-Zero", "pix2pix-zero", "000000000001",
        "Round -> square cake",
        "CLIP target +8.49 | CLIP edit +7.69 | PSNR +4.24 dB",
        "one_epoch/cycle_loss_cfg1_r32_uniform_e1_all_editors",
    ),
    Candidate(
        "L", "Pix2Pix-Zero", "pix2pix-zero", "123000000007",
        "Sofa -> bed",
        "CLIP edit +3.44 | LPIPS -0.087 | PSNR +2.21 dB",
        "one_epoch/cycle_loss_cfg1_r32_uniform_e1_all_editors",
    ),
)


FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
PANEL_SIZE = 512
LEFT_MARGIN = 390
RIGHT_MARGIN = 28
TOP_MARGIN = 92
BOTTOM_MARGIN = 28
COLUMN_GAP = 18
ROW_GAP = 92
ROWS_PER_PAGE = 4
OUR_COLOR = "#0072B2"
METRIC_COLOR = "#007C66"


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
        position, wrapped, font=selected_font, fill=fill, spacing=spacing
    )
    return draw.multiline_textbbox(
        position, wrapped, font=selected_font, spacing=spacing
    )[3]


def render_page(
    rows: tuple[Candidate, ...],
    mapping: dict,
    data_root: Path,
    evaluation_root: Path,
) -> Image.Image:
    columns = ("Source", "DirectInv", "LoRA + DirectInv (Ours)")
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
    edit_font = font(FONT_REGULAR, 24)
    metric_font = font(FONT_BOLD, 18)

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

    for row_index, row in enumerate(rows):
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

        for column, image in enumerate((source, direct, ours)):
            x = LEFT_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
            canvas.paste(image, (x, y))
            outline = OUR_COLOR if column == 2 else "#B5B5B5"
            draw.rectangle(
                (x, y, x + PANEL_SIZE - 1, y + PANEL_SIZE - 1),
                outline=outline,
                width=5 if column == 2 else 2,
            )

        text_y = y + 7
        draw.text((20, text_y), row.label, font=label_font, fill=OUR_COLOR)
        draw.text((76, text_y + 4), row.editor, font=editor_font, fill="black")
        text_y += 56
        text_y = draw_multiline(
            draw, (20, text_y), row.edit_label, edit_font,
            fill="black", width=25,
        ) + 21
        draw_multiline(
            draw, (20, text_y), row.headline_gain, metric_font,
            fill=METRIC_COLOR, width=34, spacing=7,
        )

    return canvas


def render(data_root: Path, evaluation_root: Path, output: Path) -> None:
    mapping = json.loads((data_root / "mapping_file.json").read_text())
    pages = []
    for start in range(0, len(CANDIDATES), ROWS_PER_PAGE):
        pages.append(
            render_page(
                CANDIDATES[start : start + ROWS_PER_PAGE],
                mapping,
                data_root,
                evaluation_root,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    page_paths = []
    for page_number, page in enumerate(pages, start=1):
        page_path = output.with_name(f"{output.stem}_page{page_number}.png")
        page.save(page_path, format="PNG", dpi=(300, 300), optimize=True)
        page_paths.append(page_path)

    pdf_path = output.with_suffix(".pdf")
    pages[0].save(
        pdf_path,
        format="PDF",
        resolution=300.0,
        save_all=True,
        append_images=pages[1:],
    )
    for page_path in page_paths:
        print(page_path)
    print(pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/PIE-Bench_v1")
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/directinv_lora_triplet_candidates.png"),
    )
    args = parser.parse_args()
    render(args.data_root, args.evaluation_root, args.output)


if __name__ == "__main__":
    main()
