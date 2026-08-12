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

# Building the shared 32 kHz reference cache here means the array tasks all find it complete
# instead of each resampling 696 files. It is purely an optimisation: ensure_resampled stages
# and renames atomically, so the tasks are correct either way. Never block submission on it.
#
# .venv_eval/bin/python points into the EasyBuild Python module, so it is unusable until that
# module is loaded, and `module` is not defined in a non-interactive shell.
echo "==> preparing the shared reference cache (once, before the array)"
if ! .venv_eval/bin/python -V >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do
    [ -f "$init" ] && . "$init" && break
  done
  command -v module >/dev/null 2>&1 && module load Python/3.10.4-GCCcore-11.3.0 >/dev/null 2>&1
fi

if .venv_eval/bin/python -V >/dev/null 2>&1; then
  PYTHONPATH="$AUDIO_ROOT:$AUDIO_ROOT/editing/AudioEditingCode" .venv_eval/bin/python - <<'PYCODE'
from pathlib import Path

from editing.AudioEditingCode.code.env import PATH_LOWER_BOUND_MEDLEY
from editing.eval_medley import ensure_resampled

out = ensure_resampled(Path(PATH_LOWER_BOUND_MEDLEY), 32000)
print(f"    reference cache ready: {out} ({len(list(out.glob('*.wav')))} wavs)")
PYCODE
else
  echo "    SKIPPED: .venv_eval interpreter unusable here (run 'module load"
  echo "    Python/3.10.4-GCCcore-11.3.0' first to enable it). Submitting anyway -- each task"
  echo "    will build the cache itself, which is safe but repeats ~696 resamples per task."
fi

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "submitting from: $AUDIO_ROOT"

sbatch \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_eval_medleymd.sh
