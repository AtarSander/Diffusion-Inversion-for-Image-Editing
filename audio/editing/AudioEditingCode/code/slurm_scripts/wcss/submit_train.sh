#!/bin/bash
# ABOUTME: Source audio/.env, verify the SLURM account/partition are actually set, then submit
# ABOUTME: the inversion-LoRA training array. Guards against empty vars silently choosing a CPU node.
#
# Usage (from audio/):  bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh
# Extra sbatch flags are passed through, e.g. ... --array=0-11 to run only the first config.

set -euo pipefail

# wcss -> slurm_scripts -> code -> AudioEditingCode -> editing -> audio
AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$AUDIO_ROOT"

if [ ! -f .env ]; then
  echo "ERROR: $AUDIO_ROOT/.env not found. Copy the WCSS block from .env.example." >&2
  exit 1
fi
set -a; source .env; set +a

# An unset or empty variable expands to "" and sbatch then lets WCSS's automatic partition
# chooser pick by walltime alone -- which ignores --gres and lands on a CPU partition. ${VAR:?}
# aborts on both unset and empty, so that cannot happen silently.
: "${HPC_PWR_ACCOUNT:?not set or empty in .env}"
: "${HPC_PWR_PARTITION:?not set or empty in .env}"
: "${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}"

# SLURM will not create the log directory, and tasks die without output if it is missing.
mkdir -p outputs/logs/slurm

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "submitting from: $AUDIO_ROOT"

sbatch \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_train_inversion_lora.sh
