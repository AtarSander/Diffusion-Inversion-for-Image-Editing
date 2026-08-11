#!/bin/bash
# ABOUTME: Build the MedleyMD metrics environment, including the two packages that must be
# ABOUTME: installed with --no-deps, and verify every import the eval actually uses.
#
# Usage (from audio/):  bash editing/install_eval_env.sh [venv_dir]

set -euo pipefail

AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AUDIO_ROOT"
VENV="${1:-.venv_eval}"

AUDIOLDM_EVAL_URL="https://github.com/haoheliu/audioldm_eval/archive/8dc07ee7c42f9dc6e295460a1034175a0d49b436.tar.gz"
CLAP_CKPT="res/clap/pretrained/music_audioset_epoch_15_esc_90.14.pt"
CLAP_URL="https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

if [ ! -d "$VENV" ]; then
  echo "==> creating $VENV"
  python -m venv "$VENV"
fi
PY="$VENV/bin/python"

echo "==> installing pinned dependencies"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements_eval.txt

# These two must bypass dependency resolution:
#  * audioldm_eval depends on ssr_eval, which declares PyPI `wave` (it means the stdlib module);
#    `wave` pulls MySQL-python, whose setup.py imports ConfigParser and dies on Python 3.
#  * the git+ URL in requirements.txt cannot clone: .gitmodules has no url for src/hear21passt,
#    so the commit tarball is used instead.
echo "==> installing ssr_eval and audioldm_eval with --no-deps"
"$PY" -m pip install -q --no-deps "ssr_eval==0.0.7" "audioldm_eval @ ${AUDIOLDM_EVAL_URL}"

if [ ! -f "$CLAP_CKPT" ]; then
  echo "==> downloading CLAP checkpoint (2.4 GB)"
  mkdir -p "$(dirname "$CLAP_CKPT")"
  curl -fL -o "${CLAP_CKPT}.part" "$CLAP_URL"
  mv "${CLAP_CKPT}.part" "$CLAP_CKPT"
fi

echo "==> verifying imports"
PYTHONPATH="$AUDIO_ROOT:$AUDIO_ROOT/editing/AudioEditingCode" "$PY" - <<'PYCODE'
import importlib
import sys

checks = [
    ("audioldm_eval", "calculate_fid"),
    ("audioldm_eval.datasets.load_mel", "MelPairedDataset"),
    ("audioldm_eval.feature_extractors.panns", "Cnn14"),
    ("audioldm_eval.metrics.fad", "FrechetAudioDistance"),
    ("ssr_eval.metrics", "AudioMetrics"),
    ("skimage.metrics", "structural_similarity"),
    ("laion_clap", None),
    ("muq", "MuQMuLan"),
    ("editing.eval_medley", "main"),
    ("src.metrics.alignment", "MusicAlignmentEval"),
]
bad = []
for module, attr in checks:
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        print(f"  OK  {module}" + (f".{attr}" if attr else ""))
    except Exception as exc:
        print(f"  FAIL {module}: {type(exc).__name__}: {exc}")
        bad.append(module)
if bad:
    sys.exit(f"\n{len(bad)} import(s) failed: {bad}")
print("\nall eval imports resolve")
PYCODE

echo
echo "Done. Run the eval with:"
echo "  PYTHONPATH=.:editing/AudioEditingCode $PY -m editing.eval_medley --path_audio <run>/audios"
