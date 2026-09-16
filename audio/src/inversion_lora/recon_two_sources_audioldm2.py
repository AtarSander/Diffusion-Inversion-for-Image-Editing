# ABOUTME: AudioLDM2 real-audio reconstruction on two sources (MusicCaps train audio vs MedleyDB
# ABOUTME: benchmark), for DDIM no-LoRA / trajectory-LoRA / realfn-LoRA. The SAO recon control.

import json
import sys
from pathlib import Path

import fire
import pandas as pd
import torch
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_EDITING_CODE = AUDIO_ROOT / "editing/AudioEditingCode/code"
for p in (AUDIO_ROOT, AUDIO_EDITING_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv

from models import load_model
from src.inversion_lora.apply_lora import attach_inversion_lora
from src.inversion_lora.generate_real_pairs_audioldm2 import encode_clip
from src.inversion_lora.generate_trajectories import latent_height
from src.inversion_lora.reconstruct import (
    batch_latents,
    reconstruction_metrics,
)


def musiccaps_source(audio_dir: Path, captions_csv: Path, count: int):
    """First `count` MusicCaps clips present on disk, with their captions."""
    meta = pd.read_csv(captions_csv)
    paths, caps = [], []
    for _, row in meta.iterrows():
        wav = audio_dir / f"[{row.ytid}]-[{int(row.start_s)}-{int(row.end_s)}].wav"
        if wav.exists():
            paths.append(wav)
            caps.append(str(row.caption))
        if len(paths) >= count:
            break
    return paths, caps


def medleydb_source(audio_root: Path, prompts_csv: Path, count: int):
    """First `count` distinct MedleyDB tracks, with their source captions."""
    frame = pd.read_csv(prompts_csv, index_col=0).drop_duplicates(subset="filename").head(count)
    paths, caps = [], []
    for _, row in frame.iterrows():
        name = str(row["filename"])
        paths.append(audio_root / name.split("_MIX")[0] / name)
        caps.append(str(row["source_captions"]))
    return paths, caps


def main(traj_ckpt: str, realfn_ckpt: str, count: int = 35, batch_size: int = 4,
         out_root: str = "output/recon_sources_audioldm2") -> None:
    """Score DDIM reconstruction on MusicCaps and MedleyDB for three adapter states.

    Args:
        traj_ckpt: Trajectory-trained AudioLDM2 adapter checkpoint (.pt, with .json sidecar).
        realfn_ckpt: Real-audio forward-noise adapter checkpoint.
        count: Clips/tracks per source.
        batch_size: Latents per reconstruction batch.
        out_root: Destination, relative to `audio/`.
    """
    from datetime import datetime

    load_dotenv(AUDIO_ROOT / ".env", override=True)
    import os

    device = torch.device("cuda:0")
    ldm = load_model("cvssp/audioldm2-large", device, 200, edit_method="ddim")
    ldm.model.unet.eval()
    frames = latent_height(ldm.model, 10.24)

    sources = {
        "MusicCaps (train audio)": musiccaps_source(
            Path(os.environ["MUSICCAPS_AUDIO_DIR"]),
            AUDIO_ROOT / "editing/music_caps_dl/metadata/musiccaps-public.csv", count),
        "MedleyDB (benchmark)": medleydb_source(
            Path(os.environ["MEDLEYDB_AUDIO_DIR"]),
            AUDIO_ROOT / "editing/AudioEditingCode/MedleyMDPrompts/captions_gpt5.csv", count),
    }

    encoded = {}
    for name, (paths, caps) in sources.items():
        lat = [encode_clip(ldm, p, frames).to(device) for p in paths]
        encoded[name] = (batch_latents(lat, batch_size), caps)
        logger.info("{}: {} clips, latent {}", name, len(lat), tuple(lat[0].shape[1:]))

    # Arms share the eval; the adapter is attached once and toggled. no-LoRA passes None.
    results = {name: {} for name in sources}
    arms = [("no LoRA", None), ("trajectory LoRA", traj_ckpt), ("realfn LoRA", realfn_ckpt)]
    for arm, ckpt in arms:
        set_enabled = None
        if ckpt is not None:
            # Fresh model per adapter: the injected name is reused, so re-attaching conflicts.
            ldm = load_model("cvssp/audioldm2-large", device, 200, edit_method="ddim")
            ldm.model.unet.eval()
            set_enabled = attach_inversion_lora(ldm.model.unet, ckpt)
        for name, (batches, caps) in encoded.items():
            metrics, _ = reconstruction_metrics(ldm, batches, caps, set_enabled)
            results[name][arm] = metrics
            logger.info("{} | {}: mel_psnr={:.2f} latent_mse={:.4f}",
                        name, arm, metrics["mel_psnr"], metrics["latent_mse"])

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({name: {arm: round(results[name][arm]["mel_psnr"], 2)
                                 for arm, _ in arms} for name in sources})
    table.to_csv(out / "recon_psnr.csv")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(
        "# AudioLDM2 real-audio reconstruction, two sources (mel PSNR dB)\n\n"
        + table.to_markdown() + "\n")
    print(table.to_string())
    logger.success("wrote {}", out)


if __name__ == "__main__":
    fire.Fire(main)
