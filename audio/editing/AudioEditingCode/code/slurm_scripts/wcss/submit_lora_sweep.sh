#!/bin/bash
# ABOUTME: Source audio/.env, check the SLURM account, the checkpoints and the paired reference,
# ABOUTME: print the grid, then submit the inversion-LoRA arm of the hparam sweep.
#
# Usage (from audio/):  bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_lora_sweep.sh
# Extra sbatch flags pass through, e.g. --array=0-7 for the first checkpoint only.

set -euo pipefail

# wcss -> slurm_scripts -> code -> AudioEditingCode -> editing -> audio
AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$AUDIO_ROOT"

if [ ! -f .env ]; then
  echo "ERROR: $AUDIO_ROOT/.env not found. Copy the WCSS block from .env.example." >&2
  exit 1
fi
set -a; source .env; set +a

: "${HPC_PWR_ACCOUNT:?not set or empty in .env}"
: "${HPC_PWR_PARTITION:?not set or empty in .env}"
: "${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}"

mkdir -p outputs/logs/slurm

source "editing/AudioEditingCode/code/slurm_scripts/wcss/lora_sweep_configs.sh"
mapfile -t CONFIGS < <(lora_sweep_configs)

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "split     : $LORA_SPLIT"
echo "grid      : ${#LORA_CHECKPOINTS[@]} checkpoints x ${#LORA_TSTART[@]} tstart x ${#LORA_CFG_TAR[@]} cfg_tar = ${#CONFIGS[@]} runs"

# Check every checkpoint here, on the login node, rather than failing 48 tasks one by one.
missing=0
for ckpt in "${LORA_CHECKPOINTS[@]}"; do
  path="$LORAINV_CHECKPOINT_ROOT/$ckpt"
  if [ ! -f "$path" ]; then
    echo "  MISSING checkpoint: $path" >&2
    missing=1
  elif [ ! -f "${path%.pt}.json" ]; then
    echo "  MISSING sidecar: ${path%.pt}.json (carries the LoRA config)" >&2
    missing=1
  else
    echo "  ok  $ckpt"
  fi
done
[ "$missing" -eq 0 ] || exit 1

for i in "${!CONFIGS[@]}"; do
  IFS='|' read -r ckpt tstart cfg <<< "${CONFIGS[$i]}"
  printf '  %2d  tstart=%-3s cfg_tar=%-4s  %s\n' "$i" "$tstart" "$cfg" \
    "$(lora_sweep_run_name "$ckpt" "$tstart" "$cfg")"
done

REF_DIR="$(python3 -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env; print(env.medley_split_paths("'"$LORA_SPLIT"'")[1])' 2>/dev/null || true)"
if [ -n "$REF_DIR" ]; then
  echo "reference : $(find "$REF_DIR" -name 'a*.wav' 2>/dev/null | wc -l) wavs in $REF_DIR"
fi

EXPORT_ARGS=()
if [ -n "${SKIP_EXISTING:-}" ]; then
  echo "skip_existing: on (resuming)"
  EXPORT_ARGS=(--export=ALL,SKIP_EXISTING="$SKIP_EXISTING")
fi

sbatch \
  "${EXPORT_ARGS[@]}" \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_lora_sweep.sh
