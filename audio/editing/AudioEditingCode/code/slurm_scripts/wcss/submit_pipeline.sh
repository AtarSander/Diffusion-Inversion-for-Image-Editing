#!/bin/bash
# ABOUTME: Queue the whole inversion-LoRA pipeline as chained SLURM jobs, so nothing has to be
# ABOUTME: run from the login node and each stage only starts if the previous one succeeded.
#
# Usage (from audio/):
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_pipeline.sh [stage]
#
#   stage = all       (default) setup -> trajectories -> verify -> train
#           nosetup   trajectories -> verify -> train, when the venv is already built
#           train     verify -> train, when the dataset already exists
#
# Every stage waits on the previous one with --dependency=afterok, so a failure stops the chain
# instead of training on a dataset that was never produced.

set -euo pipefail

# wcss -> slurm_scripts -> code -> AudioEditingCode -> editing -> audio
AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$AUDIO_ROOT"
HERE="editing/AudioEditingCode/code/slurm_scripts/wcss"

if [ ! -f .env ]; then
  echo "ERROR: $AUDIO_ROOT/.env not found. Copy the WCSS block from .env.example." >&2
  exit 1
fi
set -a; source .env; set +a

: "${HPC_PWR_ACCOUNT:?not set or empty in .env}"
: "${HPC_PWR_PARTITION:?not set or empty in .env}"
: "${LORAINV_DATA_ROOT:?not set or empty in .env}"
: "${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}"

if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "WARNING: no W&B credential found. Add WANDB_API_KEY to ~/.bashrc, or the training" >&2
  echo "         stage will refuse to start." >&2
fi

mkdir -p outputs/logs/slurm

STAGE="${1:-all}"
SUBMIT=(sbatch --account="$HPC_PWR_ACCOUNT" --partition="$HPC_PWR_PARTITION")

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "data root : $LORAINV_DATA_ROOT"
echo "stage     : $STAGE"
echo

previous=""
queue() {
  # queue <label> <script> [extra sbatch flags...]
  local label="$1" script="$2"; shift 2
  local args=("${SUBMIT[@]}")
  if [ -n "$previous" ]; then args+=(--dependency=afterok:"$previous"); fi
  args+=("$@" "$HERE/$script")
  local job_id
  job_id="$("${args[@]}" | awk '{print $NF}')"
  printf '%-14s job %s%s\n' "$label" "$job_id" "${previous:+  (after $previous)}"
  previous="$job_id"
}

case "$STAGE" in
  all)     queue setup run_setup_env.sh ;&
  nosetup) queue trajectories run_generate_trajectories.sh ;&
  train)   queue verify run_verify_trajectories.sh
           queue train run_train_inversion_lora.sh ;;
  *) echo "unknown stage: $STAGE (expected all, nosetup or train)" >&2; exit 2 ;;
esac

echo
echo "Watch with:  squeue -u \$USER"
echo "Logs land in outputs/logs/slurm/"
