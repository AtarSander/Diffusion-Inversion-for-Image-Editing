#!/bin/bash
# ABOUTME: Source audio/.env, verify the SLURM account/partition are actually set, then submit
# ABOUTME: the MedleyMD eval array. Guards against empty vars silently choosing a CPU node.
#
# Usage (from audio/):  bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_baselines.sh
# Extra sbatch flags are passed through, e.g. ... --array=0-11 to run only the first config.

set -euo pipefail

# wcss -> slurm_scripts -> code -> AudioEditingCode -> editing -> audio
AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$AUDIO_ROOT"

if [ ! -f .env ]; then
  echo "ERROR: $AUDIO_ROOT/.env not found. Copy the WCSS block from .env.example." >&2
  exit 1
fi
unset EDIT_OUTPUTS_DIR MEDLEY_LOWER_BOUND_DIR ALDM2_TEMP_DIR MEDLEYDB_AUDIO_DIR
set -a; source .env; set +a

# An unset or empty variable expands to "" and sbatch then lets WCSS's automatic partition
# chooser pick by walltime alone -- which ignores --gres and lands on a CPU partition. ${VAR:?}
# aborts on both unset and empty, so that cannot happen silently.
: "${HPC_PWR_ACCOUNT:?not set or empty in .env}"
: "${HPC_PWR_PARTITION:?not set or empty in .env}"

# SLURM will not create the log directory, and tasks die without output if it is missing.
mkdir -p outputs/logs/slurm

# All array tasks share the reference directory, so if they each try to build its 32 kHz cache
# they race: one creates the directory and the rest read it half filled. Build it once here,
# serially, so every task finds it complete. ensure_resampled is idempotent.
echo "==> preparing the shared reference cache (once, before the array)"
PYTHONPATH="$AUDIO_ROOT:$AUDIO_ROOT/editing/AudioEditingCode" .venv_eval/bin/python - <<'PYCODE'
from pathlib import Path

from editing.AudioEditingCode.code.env import PATH_LOWER_BOUND_MEDLEY
from editing.eval_medley import ensure_resampled

out = ensure_resampled(Path(PATH_LOWER_BOUND_MEDLEY), 32000)
print(f"    reference cache ready: {out} ({len(list(out.glob('*.wav')))} wavs)")
PYCODE

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "submitting from: $AUDIO_ROOT"

sbatch \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_eval_medleymd.sh
