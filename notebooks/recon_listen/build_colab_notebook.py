# ABOUTME: Build a single self-contained Colab notebook with base64-embedded mp3 audio players
# ABOUTME: comparing real-audio reconstructions (source vs four inversion methods).

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MP3 = HERE / "mp3"

EXAMPLES = {
    "MedleyDB — benchmark audio (LoRA was NOT trained on this)": ["medley_a12", "medley_a129"],
    "MusicCaps — train audio (LoRA WAS trained on this distribution)": ["musiccaps_a81",
                                                                        "musiccaps_a48"],
}
METHODS = [("source", "Source (reference)"), ("ddpm", "DDPM Inv."), ("nolora", "ODE Inv."),
           ("traj", "ODE Inv. w/ LoRA"), ("realfn", "ODE Inv. w/ Real-Audio-LoRA")]
PSNR = {
    "medley_a12": {"ddpm": 22.4, "nolora": 19.8, "traj": 21.1, "realfn": 17.7},
    "medley_a129": {"ddpm": 21.6, "nolora": 21.2, "traj": 21.3, "realfn": 13.9},
    "musiccaps_a81": {"ddpm": 25.7, "nolora": 26.2, "traj": 26.0, "realfn": 26.3},
    "musiccaps_a48": {"ddpm": 27.1, "nolora": 26.6, "traj": 27.2, "realfn": 27.3},
}


def b64(stem: str, method: str) -> str:
    """Base64-encode one mp3, or empty string if absent."""
    p = MP3 / f"{stem}_{method}.mp3"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def build_html() -> str:
    """One HTML block: a titled section per source, a labelled audio row per clip."""
    css = (
        "<style>"
        ".rec{font-family:system-ui,sans-serif;max-width:760px}"
        ".rec h2{margin:1.2em 0 .3em;border-bottom:2px solid #333;padding-bottom:.2em}"
        ".rec .clip{margin:.8em 0 1.4em}.rec .clip b{font-size:1.05em}"
        ".rec .row{display:flex;align-items:center;gap:10px;margin:.25em 0}"
        ".rec .lab{width:230px;font-size:.9em}.rec .psnr{color:#666;font-size:.85em}"
        ".rec .realfn .lab{color:#c0392b;font-weight:600}"
        ".rec audio{height:32px}</style>"
    )
    out = [css, "<div class='rec'>",
           "<h1>Real-audio reconstruction — listen</h1>",
           "<p>Each row: invert the source, denoise back, compare to the source. "
           "The <b>Real-Audio-LoRA</b> was trained only on MusicCaps: on its own training-audio "
           "distribution it matches the other methods (all four within ~0.3 dB below), but it "
           "degrades sharply on the MedleyDB benchmark it never saw.</p>"]
    for title, stems in EXAMPLES.items():
        out.append(f"<h2>{title}</h2>")
        for stem in stems:
            out.append(f"<div class='clip'><b>{stem}</b>")
            for method, label in METHODS:
                data = b64(stem, method)
                if not data:
                    continue
                cls = "row realfn" if method == "realfn" else "row"
                psnr = "" if method == "source" else \
                    f"<span class='psnr'>PSNR {PSNR[stem][method]:.1f} dB</span>"
                out.append(
                    f"<div class='{cls}'><span class='lab'>{label} {psnr}</span>"
                    f"<audio controls preload='none' "
                    f"src='data:audio/mp3;base64,{data}'></audio></div>"
                )
            out.append("</div>")
    out.append("</div>")
    return "".join(out)


html_block = build_html()

# The audio lives in the cell OUTPUT (rendered as a sandboxed HTML widget), not in the source.
# Putting megabytes of base64 in an editable code cell freezes Colab's editor; the output
# renderer handles it fine. Source stays tiny, so nothing to run — it's pre-rendered.
nb = {
    "cells": [{
        "cell_type": "code",
        "metadata": {},
        "execution_count": 1,
        "outputs": [{"output_type": "display_data", "metadata": {},
                     "data": {"text/html": html_block, "text/plain": ["<audio players>"]}}],
        "source": ("# Real-audio reconstruction comparison — the players are rendered below.\n"
                   "# Pre-rendered and self-contained; nothing to run.\n"),
    }],
    "metadata": {"colab": {"name": "reconstruction_comparison"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 0,
}
out = HERE / "lorainv_20260916_reconstruction_comparison.ipynb"
out.write_text(json.dumps(nb))
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

# Standalone HTML: opens in any browser by double-click, players work offline, nothing to run.
html_out = HERE / "lorainv_20260916_reconstruction_comparison.html"
html_out.write_text("<!doctype html><meta charset='utf-8'>"
                    "<title>Reconstruction comparison</title>" + build_html())
print(f"wrote {html_out} ({html_out.stat().st_size / 1e6:.1f} MB)")
