# ABOUTME: Resolves every filesystem path the editing/eval scripts need.
# ABOUTME: Repo-relative paths derive from __file__; machine-specific data dirs come from .env.

import os
from pathlib import Path

from dotenv import load_dotenv

# .../audio/editing/AudioEditingCode/code/env.py -> .../audio
AUDIO_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(AUDIO_ROOT / ".env", override=False)


def _from_env(name: str, default: Path) -> str:
    """Return an overridable path as a string (callers concatenate these, so never Path)."""
    return str(Path(os.environ.get(name, str(default))).expanduser())


# --- Datasets (machine-specific: override in audio/.env when the repo moves servers) ---------

# MedleyDB V1 mixes, laid out as <root>/<Track>/<Track>_MIX.wav, which is what
# prepare_dataset() reconstructs via filename.split("_MIX")[0].
PATH_AUDIOS_MEDLEY = _from_env("MEDLEYDB_AUDIO_DIR", Path("/nas/lstanisz/data/medleydb/V1_mix"))

# MusicCaps clips. Not populated yet; only needed for real-audio-seeded trajectories.
PATH_MUSICCAPS = _from_env("MUSICCAPS_AUDIO_DIR", AUDIO_ROOT / "data/musiccaps/audio")

# --- Repo-relative (portable across servers, no override needed) -----------------------------

# The only prompt CSV carrying all four columns the edit drivers read:
# filename, source_captions, target_captions, edit.
PATH_PROMPTS_MEDLEY = str(
    AUDIO_ROOT / "editing/AudioEditingCode/MedleyMDPrompts/captions_gpt5.csv"
)

# Paired per-example reference for FAD and mel PSNR/SSIM, built by editing/build_lower_bound.py.
#
# MUST correspond to the same split as PATH_PROMPTS_MEDLEY above: the drivers name outputs
# a{idx}.wav by row position, so a full-set reference (a0..a695) paired with a split's edits
# (a0..a348) overlaps only partially. get_filename_intersection_ratio() then falls below its
# 0.99 threshold and calculate_psnr_ssim() returns -1 *without raising*. Change both together.
PATH_LOWER_BOUND_MEDLEY = _from_env(
    "MEDLEY_LOWER_BOUND_DIR", AUDIO_ROOT / "outputs/medleymd/lower_bound_full/audios"
)

# Edited audio lands in <PATH_EDIT_OUTPUTS>/<dataset>/<model>/<run_name>/audios/a{idx}.wav.
# Some callers do PATH_EDIT_OUTPUTS + "/medleymd", so this must stay a plain string.
PATH_EDIT_OUTPUTS = _from_env("EDIT_OUTPUTS_DIR", AUDIO_ROOT / "outputs/edits")

# Scratch for AudioLDM2's 60 s truncated copies; audioldm_run.py creates it on demand.
ALDM2_TEMP_DIR = _from_env("ALDM2_TEMP_DIR", AUDIO_ROOT / ".temp/audioldm2")
