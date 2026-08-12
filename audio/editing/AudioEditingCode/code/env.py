# ABOUTME: Resolves every filesystem path the editing/eval scripts need, from audio/.env.
# ABOUTME: .env is authoritative: it overrides the ambient environment, so nothing can drift.

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# .../audio/editing/AudioEditingCode/code/env.py -> .../audio
AUDIO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = AUDIO_ROOT / ".env"

# override=True is the point of this module: .env is the one place configuration is edited, so
# a variable left exported in a shell (or inherited by a SLURM job, which exports the submitting
# environment by default) must never win over the file. HF_TOKEN is deliberately NOT declared in
# .env -- it is a secret and comes from ~/.bashrc; keys absent from .env are left untouched here.
load_dotenv(ENV_FILE, override=True)

_DOTENV = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
_DOTENV_KEYS = set(_DOTENV)

# PROJECT_ROOT is only used to interpolate the other values inside .env, so a file copied from
# another machine still parses -- it just points every path at a layout that is not here. Compare
# it against the checkout this file actually lives in and fail at import rather than letting a
# job write to nowhere.
_declared_root = _DOTENV.get("PROJECT_ROOT")
if _declared_root and Path(_declared_root).resolve() != AUDIO_ROOT.parent.resolve():
    raise RuntimeError(
        f"{ENV_FILE} sets PROJECT_ROOT={_declared_root}, but this checkout is "
        f"{AUDIO_ROOT.parent}. Every ${{PROJECT_ROOT}} path in .env would point elsewhere. "
        "Update PROJECT_ROOT to match this machine."
    )


def _resolve(name: str, default: Path) -> str:
    """Return a configured path as a string (callers concatenate these, so never Path)."""
    raw = os.environ.get(name)
    return str(Path(raw).expanduser()) if raw else str(default)


def source_of(name: str) -> str:
    """Report where a setting came from, so a surprising value can be traced quickly."""
    if name in _DOTENV_KEYS:
        return ".env"
    return "shell" if name in os.environ else "default"


# --- Datasets --------------------------------------------------------------------------------

# MedleyDB V1 mixes, laid out as <root>/<Track>/<Track>_MIX.wav, which is what
# prepare_dataset() reconstructs via filename.split("_MIX")[0].
PATH_AUDIOS_MEDLEY = _resolve("MEDLEYDB_AUDIO_DIR", AUDIO_ROOT / "data/medleydb/V1_mix")

# MusicCaps clips; only needed for real-audio-seeded inversion trajectories.
PATH_MUSICCAPS = _resolve("MUSICCAPS_AUDIO_DIR", AUDIO_ROOT / "data/musiccaps/audio")

# The only prompt CSV carrying all four columns the edit drivers read:
# filename, source_captions, target_captions, edit.
PATH_PROMPTS_MEDLEY = _resolve(
    "MEDLEY_PROMPTS_CSV",
    AUDIO_ROOT / "editing/AudioEditingCode/MedleyMDPrompts/captions_gpt5.csv",
)

# --- Outputs ---------------------------------------------------------------------------------

# Edited audio lands in <PATH_EDIT_OUTPUTS>/<dataset>/<model>/<run_name>/audios/a{idx}.wav.
# Some callers do PATH_EDIT_OUTPUTS + "/medleymd", so this must stay a plain string.
PATH_EDIT_OUTPUTS = _resolve("EDIT_OUTPUTS_DIR", AUDIO_ROOT / "outputs/edits")

# Paired per-example reference for FAD and mel PSNR/SSIM, built by editing/build_lower_bound.py.
# It is a copy of each row's input mix, renamed a{idx}.wav so the paired metrics can align by
# filename. MUST correspond to the same split as PATH_PROMPTS_MEDLEY: outputs are named by row
# position, so pairing a full reference with a split's edits drops the filename intersection
# below get_filename_intersection_ratio's 0.99 threshold and calculate_psnr_ssim returns -1
# without raising.
PATH_LOWER_BOUND_MEDLEY = _resolve(
    "MEDLEY_LOWER_BOUND_DIR", AUDIO_ROOT / "outputs/medleymd/lower_bound_full/audios"
)

# Scratch for AudioLDM2's 60 s truncated copies; audioldm_run.py creates it on demand.
ALDM2_TEMP_DIR = _resolve("ALDM2_TEMP_DIR", AUDIO_ROOT / ".temp/audioldm2")

SETTINGS = {
    "PATH_AUDIOS_MEDLEY": ("MEDLEYDB_AUDIO_DIR", PATH_AUDIOS_MEDLEY),
    "PATH_PROMPTS_MEDLEY": ("MEDLEY_PROMPTS_CSV", PATH_PROMPTS_MEDLEY),
    "PATH_MUSICCAPS": ("MUSICCAPS_AUDIO_DIR", PATH_MUSICCAPS),
    "PATH_EDIT_OUTPUTS": ("EDIT_OUTPUTS_DIR", PATH_EDIT_OUTPUTS),
    "PATH_LOWER_BOUND_MEDLEY": ("MEDLEY_LOWER_BOUND_DIR", PATH_LOWER_BOUND_MEDLEY),
    "ALDM2_TEMP_DIR": ("ALDM2_TEMP_DIR", ALDM2_TEMP_DIR),
}


if __name__ == "__main__":
    print(f".env: {ENV_FILE}{'' if ENV_FILE.exists() else '  (MISSING)'}")
    for name, (var, value) in SETTINGS.items():
        print(f"  {name:26s} [{source_of(var):7s}] {value}")
