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
# Reconstruction mel PSNR on each source (from output/recon_sources), for reference.
PSNR = {"MedleyDB (benchmark)": {"ddpm": 23.3, "nolora": 22.2, "traj": 22.3, "realfn": 15.6},
        "MusicCaps (train audio)": {"ddpm": 23.7, "nolora": 23.0, "traj": 23.2, "realfn": 23.0}}
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
    src = next(s for s, stems in EXAMPLES.items() if stem in stems)
    tag = "" if method == "source" else f"  —  recon PSNR {PSNR[src][method]:.1f} dB"
    print(f"{METHODS[method]}{tag}")
    ipd.display(ipd.Audio(str(path)))


for source, stems in EXAMPLES.items():
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    for stem in stems:
        print(f"\n--- {stem} ---")
        for method in METHODS:
            player(stem, method)


# %%
# The headline A/B: realfn on MedleyDB (collapses to 15.6 dB) vs on MusicCaps (fine, ~23 dB).
print("MedleyDB a12 — no-LoRA (22.2) vs Real-Audio-LoRA (15.6):")
for m in ["nolora", "realfn"]:
    player("medley_a12", m)
print("\nMusicCaps a0 — same two, both ~23 dB (realfn fine in-distribution):")
for m in ["nolora", "realfn"]:
    player("musiccaps_a0", m)
