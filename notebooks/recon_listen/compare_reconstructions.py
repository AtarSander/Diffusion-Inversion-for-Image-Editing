# ABOUTME: Jupyter-style A/B listening notebook for real-audio reconstructions: source vs the four
# ABOUTME: inversion methods, on two MedleyDB (benchmark) and two MusicCaps (train) examples.

# %%
# Parameters
from pathlib import Path

import IPython.display as ipd

AUDIO_DIR = Path(__file__).resolve().parent / "audio" if "__file__" in dir() else Path("audio")
EXAMPLES = {
    "MedleyDB (benchmark)": ["medley_a12", "medley_a129"],
    "MusicCaps (train audio)": ["musiccaps_a0", "musiccaps_a1"],
}
# Per-CLIP reconstruction mel PSNR (dB) for exactly the examples below — not the split average,
# which washes out realfn's per-clip variance (its 180-clip mean ties no-LoRA while individual
# clips like a0/a1 sit well below it).
PSNR = {
    "medley_a12": {"ddpm": 22.4, "nolora": 19.8, "traj": 21.1, "realfn": 17.7},
    "medley_a129": {"ddpm": 21.6, "nolora": 21.2, "traj": 21.3, "realfn": 13.9},
    "musiccaps_a0": {"ddpm": 26.7, "nolora": 26.0, "traj": 26.6, "realfn": 22.1},
    "musiccaps_a1": {"ddpm": 22.6, "nolora": 22.9, "traj": 20.5, "realfn": 20.7},
}
METHODS = {"source": "source (reference)", "ddpm": "DDPM Inv.", "nolora": "ODE Inv.",
           "traj": "ODE Inv. w/ LoRA", "realfn": "ODE Inv. w/ Real-Audio-LoRA"}


# %%
# Play every example: source first, then the four methods. Run in Jupyter / VS Code interactive
# to get inline audio players. IPython.display.Audio(path) needs no audio-loading library.
def player(stem: str, method: str):
    """Show a labelled inline audio player for one reconstruction, if the file exists."""
    path = AUDIO_DIR / f"{stem}_{method}.wav"
    if not path.exists():
        return
    tag = "" if method == "source" else f"  —  this clip's PSNR {PSNR[stem][method]:.1f} dB"
    print(f"{METHODS[method]}{tag}")
    ipd.display(ipd.Audio(str(path)))


for source, stems in EXAMPLES.items():
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    for stem in stems:
        print(f"\n--- {stem} ---")
        for method in METHODS:
            player(stem, method)


# %%
# The headline A/B: no-LoRA vs Real-Audio-LoRA, per clip. realfn is below no-LoRA on every clip
# here (worse in-distribution too), and collapses hardest on MedleyDB (a129: 21.2 -> 13.9 dB).
print("MedleyDB a129 — no-LoRA (21.2) vs Real-Audio-LoRA (13.9):")
for m in ["nolora", "realfn"]:
    player("medley_a129", m)
print("\nMusicCaps a0 — no-LoRA (26.0) vs Real-Audio-LoRA (22.1):")
for m in ["nolora", "realfn"]:
    player("musiccaps_a0", m)

# %%
