# ABOUTME: Builds a self-contained HTML contact sheet for the CelebA face-steering
# ABOUTME: runs: 3 attributes x 2 prompts x {DDIM, MW} rows, alpha -3..+3 columns.
import base64
import io
import time
from pathlib import Path

from PIL import Image

ROOT = Path(".")
RUNS = {
    "Smiling": ROOT / "output/steering/celeba_smiling/20261026_024550/generate",
    "Eyeglasses": ROOT / "output/steering/celeba_eyeglasses/20261026_024550/generate",
    "Young": ROOT / "output/steering/celeba_young/20261026_024550/generate",
}
PROMPTS = {"prompt_00": "portrait photo", "prompt_08": "close-up photo"}
ALPHAS = [-3.0, -2.0, -1.0, None, 1.0, 2.0, 3.0]
THUMB = 240
TS = time.strftime("%Y%m%d_%H%M%S")


def thumb_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    head = "".join(f"<th>{'base' if a is None else f'α={a:+.0f}'}</th>" for a in ALPHAS)
    parts = [
        "<meta charset='utf-8'><title>CelebA steering contact sheet</title>",
        "<style>body{font-family:sans-serif;background:#faf9f5;color:#191918;margin:16px}"
        "table{border-collapse:collapse}td,th{padding:2px 3px;font-size:12px;text-align:center}"
        "img{width:160px;display:block}th.rowh{text-align:right;padding-right:8px;white-space:nowrap}"
        "h2{margin:18px 0 6px}</style>",
        f"<h1>CelebA-HQ face steering — contact sheet ({TS})</h1>",
        "<p>Rows: inversion method; columns: steering strength α (base = unsteered). seed 42.</p>",
    ]
    for attr, gen in RUNS.items():
        for prompt_dir, prompt_label in PROMPTS.items():
            parts.append(f"<h2>{attr} — {prompt_label} ({prompt_dir})</h2>")
            parts.append(f"<table><tr><th></th>{head}</tr>")
            for method in ("DDIM", "MW"):
                cells = []
                for a in ALPHAS:
                    name = "base.png" if a is None else f"SA_{method}_true_{a:.1f}.png"
                    path = gen / "seed_42" / prompt_dir / name
                    cells.append(f"<td><img src='data:image/jpeg;base64,{thumb_b64(path)}'></td>")
                parts.append(f"<tr><th class='rowh'>{method}</th>{''.join(cells)}</tr>")
            parts.append("</table>")
    out = ROOT / f"output/steering/celeba_summary/{TS}_celeba_contact_sheet.html"
    out.write_text("".join(parts))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
