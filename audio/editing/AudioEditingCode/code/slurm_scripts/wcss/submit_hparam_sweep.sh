#!/bin/bash
# ABOUTME: Source audio/.env, check the SLURM account/partition and the 115-row paired reference,
# ABOUTME: print the grid, then submit the AudioLDM2 hparam sweep array.
#
# Usage (from audio/):  bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_hparam_sweep.sh
# Extra sbatch flags pass through, e.g. --array=0-15 for the ddpm block only.

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

mkdir -p outputs/logs/slurm

source "editing/AudioEditingCode/code/slurm_scripts/wcss/hparam_sweep_configs.sh"
mapfile -t CONFIGS < <(sweep_configs)

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "split     : $SWEEP_SPLIT"
echo "grid      : ${#SWEEP_MODES[@]} modes x ${#SWEEP_TSTART[@]} tstart x ${#SWEEP_CFG_TAR[@]} cfg_tar = ${#CONFIGS[@]} runs"
for i in "${!CONFIGS[@]}"; do
  IFS='|' read -r mode tstart cfg <<< "${CONFIGS[$i]}"
  printf '  %2d  %-6s tstart=%-3s cfg_tar=%-4s  %s\n' \
    "$i" "$mode" "$tstart" "$cfg" "$(sweep_run_name "$mode" "$tstart" "$cfg")"
done

# The paired reference has to exist before the eval, but check it now: it is cheap to build and
# expensive to discover missing after 48 edit jobs have run.
REF_DIR="$(python3 -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env; print(env.medley_split_paths("'"$SWEEP_SPLIT"'")[1])' 2>/dev/null || true)"
if [ -n "$REF_DIR" ]; then
  ref_n=$(find "$REF_DIR" -name 'a*.wav' 2>/dev/null | wc -l)
  echo "reference : $ref_n wavs in $REF_DIR"
  if [ "$ref_n" -eq 0 ]; then
    echo "  WARNING: empty. The eval will fail until you run:" >&2
    echo "    python -m editing.build_lower_bound --splits $SWEEP_SPLIT" >&2
  fi
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
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_hparam_sweep.sh
