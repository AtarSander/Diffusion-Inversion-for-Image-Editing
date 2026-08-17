#!/bin/bash
# ABOUTME: Source audio/.env, verify the SLURM account/partition are actually set, then submit
# ABOUTME: the MedleyMD baseline array. Guards against empty vars silently choosing a CPU node.
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
set -a; source .env; set +a

# An unset or empty variable expands to "" and sbatch then lets WCSS's automatic partition
# chooser pick by walltime alone -- which ignores --gres and lands on a CPU partition. ${VAR:?}
# aborts on both unset and empty, so that cannot happen silently.
: "${HPC_PWR_ACCOUNT:?not set or empty in .env}"
: "${HPC_PWR_PARTITION:?not set or empty in .env}"

# SLURM will not create the log directory, and tasks die without output if it is missing.
mkdir -p outputs/logs/slurm

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "submitting from: $AUDIO_ROOT"

# LORA_PATH selects the LoRA config/run. sbatch does not reliably carry the submitting
# environment into the job here, so forward it explicitly instead of hoping it propagates -- and
# check the checkpoint exists now, on the login node, rather than failing 12 tasks later.
EXPORT_ARGS=()
FORWARD=()
[ -n "${SKIP_EXISTING:-}" ] && FORWARD+=("SKIP_EXISTING=$SKIP_EXISTING") && echo "skip_existing: on (resuming)"
if [ -n "${LORA_PATH:-}" ]; then
  if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: LORA_PATH=$LORA_PATH does not exist" >&2
    exit 1
  fi
  if [ ! -f "${LORA_PATH%.pt}.json" ]; then
    echo "ERROR: ${LORA_PATH%.pt}.json missing; it carries the LoRA config needed to rebuild" >&2
    echo "       the adapter. Point at a checkpoint written by src/inversion_lora/train.py." >&2
    exit 1
  fi
  echo "lora path : $LORA_PATH"
  FORWARD+=("LORA_PATH=$LORA_PATH")
fi

if [ ${#FORWARD[@]} -gt 0 ]; then
  EXPORT_ARGS=(--export=ALL,"$(IFS=,; echo "${FORWARD[*]}")")
fi

sbatch \
  "${EXPORT_ARGS[@]}" \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_baselines_medleymd.sh
