# ABOUTME: Lay out MusicCaps clips as a reconstruction benchmark split (mcrecon): the _MIX
# ABOUTME: directory layout, the prompt CSV, and the paired lower-bound reference.

import sys
from pathlib import Path

import fire
import pandas as pd

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from editing.AudioEditingCode.code.env import (  # noqa: E402
    PATH_AUDIOS_MCRECON,
    PATH_MUSICCAPS,
    PATH_PROMPTS_MEDLEY,
)


def main(num: int = 100, audio_dir: str | None = None, out_root: str | None = None) -> None:
    """Build the mcrecon split from downloaded MusicCaps clips.

    One row per available clip (in caption-file order): copies the flat `[ytid]-[a-b].wav` into
    `<out_root>/mc{idx}/mc{idx}_MIX.wav` and writes `captions_mcrecon.csv` with reconstruction
    prompts (target == source). Then builds the paired lower-bound reference for PSNR/SSIM.

    Args:
        num: Leading caption rows to consider; clips missing from `audio_dir` are skipped.
        audio_dir: Flat MusicCaps clip directory (default `MUSICCAPS_AUDIO_DIR`).
        out_root: Destination for the `_MIX` layout (default `MC_RECON_AUDIO_DIR`).
    """
    audio = Path(audio_dir or PATH_MUSICCAPS)
    root = Path(out_root or PATH_AUDIOS_MCRECON)
    root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(
        AUDIO_ROOT / "editing/music_caps_dl/metadata/musiccaps-public.csv"
    ).head(num)
    rows = []
    for _, r in meta.iterrows():
        src = audio / f"[{r.ytid}]-[{int(r.start_s)}-{int(r.end_s)}].wav"
        if not src.exists():
            continue
        idx = len(rows)
        name = f"mc{idx:04d}"
        dst_dir = root / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}_MIX.wav"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        rows.append({"filename": f"{name}_MIX.wav", "source_captions": str(r.caption),
                     "target_captions": str(r.caption), "edit": "RECON", "ytid": str(r.ytid)})

    df = pd.DataFrame(rows)
    csv = Path(PATH_PROMPTS_MEDLEY).parent / "captions_mcrecon.csv"
    df.to_csv(csv)
    print(f"{len(df)} MusicCaps clips -> {root}\nwrote {csv}")

    # Paired reference set for mel PSNR/SSIM, same layout the eval reads.
    from editing.build_lower_bound import build_split
    from editing.AudioEditingCode.code.env import PATH_LOWER_BOUND_MEDLEY
    build_split("mcrecon", Path(PATH_LOWER_BOUND_MEDLEY).parents[1])


if __name__ == "__main__":
    fire.Fire(main)
