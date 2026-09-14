# ABOUTME: Synthesise the generated-input eval split: one Stable Audio clip per unique
# ABOUTME: (track, source caption) pair, plus the split CSV whose rows point at those clips.

import json
import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from editing.AudioEditingCode.code.env import (  # noqa: E402
    PATH_AUDIOS_MEDLEY,
    PATH_AUDIOS_MEDLEY_GEN,
)
from src.inversion_lora.generate_trajectories import git_sha  # noqa: E402
from src.inversion_lora.stable_audio import decode_to_audio, load_teacher  # noqa: E402


def build_gen_csv(
    split_csv: Path, real_audio_root: Path, gen_csv: Path, seed_base: int
) -> pd.DataFrame:
    """Derive the generated-input split CSV from a real-audio split CSV.

    Rows keep their original indices (the `a{idx}.wav` numbering every driver and eval uses).
    Rows sharing a (filename, source caption) pair point at one shared generated clip, mirroring
    how rows of one track share its real mix. The real track's duration and a pinned seed are
    recorded per row so generation is fully determined by this file.

    Args:
        split_csv: The real-audio split to mirror (e.g. the 115-row hparam CSV).
        real_audio_root: Root holding `<Track>/<Track>_MIX.wav`, read only for durations.
        gen_csv: Destination CSV.
        seed_base: First seed; group `g` uses `seed_base + g`. Must never collide with the
            training-trajectory seeds (42..1541).

    Returns:
        The written dataframe.
    """
    df = pd.read_csv(split_csv, index_col=0, header=0)
    groups: dict[tuple[str, str], int] = {}
    durations: dict[str, float] = {}
    rows = []
    for idx, row in df.iterrows():
        key = (row["filename"], row["source_captions"])
        if key not in groups:
            groups[key] = len(groups)
        g = groups[key]
        real = row["filename"]
        if real not in durations:
            path = real_audio_root / real.split("_MIX")[0] / real
            info = torchaudio.info(str(path))
            durations[real] = info.num_frames / info.sample_rate
        rows.append(
            {
                "filename": f"gen{g:03d}_MIX.wav",
                "source_captions": row["source_captions"],
                "target_captions": row["target_captions"],
                "edit": row["edit"],
                "real_filename": real,
                "duration_s": round(durations[real], 3),
                "seed": seed_base + g,
            }
        )
    out = pd.DataFrame(rows, index=df.index)
    out.to_csv(gen_csv)
    logger.info(
        "{}: {} rows -> {} generated clips (seeds {}..{})",
        gen_csv.name,
        len(out),
        len(groups),
        seed_base,
        seed_base + len(groups) - 1,
    )
    return out


@hydra.main(config_path="../../config", config_name="generate_eval_inputs", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the generated-input split: the CSV, and (unless csv_only) the wavs it names."""
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    real_root = Path(cfg.real_audio_root or PATH_AUDIOS_MEDLEY)
    gen_csv = Path(cfg.gen_csv)
    if cfg.csv_only:
        build_gen_csv(Path(cfg.split_csv), real_root, gen_csv, int(cfg.seed_base))
        return
    if not gen_csv.exists():
        raise FileNotFoundError(f"{gen_csv} not found; run once with csv_only=true and commit it")

    df = pd.read_csv(gen_csv, index_col=0, header=0)
    clips = df.drop_duplicates(subset="filename", keep="first").reset_index(drop=True)
    if cfg.num_groups is not None:
        clips = clips.head(int(cfg.num_groups))

    device = torch.device(str(cfg.device))
    if device.type == "cpu":
        logger.warning("Running on CPU; use only for smoke tests.")
    else:
        torch.cuda.set_device(device)

    out_root = Path(cfg.output_dir or PATH_AUDIOS_MEDLEY_GEN)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading {} ({} steps)", cfg.model_id, cfg.num_inference_steps)
    teacher = load_teacher(
        str(cfg.model_id), device, int(cfg.num_inference_steps), schedule=str(cfg.schedule)
    )
    guidance_scale = float(cfg.guidance_scale)
    uncond_audio = teacher.encode_prompt("") if guidance_scale != 1.0 else None
    sample_rate = teacher.pipe.vae.config.sampling_rate

    written = 0
    for pos, row in tqdm(list(clips.iterrows()), desc="clips"):
        name = row["filename"]
        wav_path = out_root / name.split("_MIX")[0] / name
        if wav_path.exists() and not cfg.overwrite:
            continue
        duration = min(float(row["duration_s"]), teacher.max_duration_s)
        teacher.set_duration(duration)
        text_audio = teacher.encode_prompt(str(row["source_captions"]))
        trajectory, _, _, _ = teacher.ode_trajectory(
            text_audio,
            seed=int(row["seed"]),
            progress=False,
            guidance_scale=guidance_scale,
            uncond_text_audio=uncond_audio,
        )
        audio = decode_to_audio(teacher, trajectory[-1:].to(device))[0].clamp(-1.0, 1.0)
        assert audio.shape[0] == 2 and audio.shape[1] > 0, audio.shape
        if pos == 0:
            logger.info(
                "First clip: {!r} seed={} duration={:.2f}s wav={} peak={:.3f}",
                str(row["source_captions"])[:120],
                int(row["seed"]),
                duration,
                tuple(audio.shape),
                float(audio.abs().max()),
            )
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(wav_path), audio, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16
        )
        written += 1

    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "gen_csv": str(gen_csv),
        "clips": len(clips),
        "written": written,
        "sample_rate": sample_rate,
        "git_sha": git_sha(),
    }
    with (out_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.success("{} clips present under {} ({} newly written)", len(clips), out_root, written)


if __name__ == "__main__":
    main()
