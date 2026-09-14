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

# Must be the same grid the job will read, or this preview and its checkpoint check describe a
# different sweep than the one that runs. SCRIPT is only echoed and forwarded: the job resolves it.
SWEEP_CONFIGS="${SWEEP_CONFIGS:-editing/AudioEditingCode/code/slurm_scripts/wcss/lora_sweep_configs.sh}"
source "$SWEEP_CONFIGS"
mapfile -t CONFIGS < <(lora_sweep_configs)

echo "account   : $HPC_PWR_ACCOUNT"
echo "partition : $HPC_PWR_PARTITION"
echo "grid file : $SWEEP_CONFIGS"
echo "entrypoint: ${SCRIPT:-edit_audioldm_medleydb.py}"
echo "split     : $LORA_SPLIT"
# A budget-matched grid has no LORA_TSTART: a fixed NFE pins tstart per method and varies the
# grid length instead, so it enumerates depth points directly. set -u makes the missing array
# fatal rather than cosmetic, which is why this is guarded rather than just tolerated.
if [ -n "${LORA_TSTART+x}" ]; then
  echo "grid      : ${#LORA_CHECKPOINTS[@]} checkpoints x ${#LORA_TSTART[@]} tstart x ${#LORA_CFG_TAR[@]} cfg_tar = ${#CONFIGS[@]} runs"
else
  echo "grid      : ${#CONFIGS[@]} runs from ${#LORA_CHECKPOINTS[@]} checkpoint(s), depth points at a fixed NFE budget"
fi

# Check every checkpoint here, on the login node, rather than failing 48 tasks one by one.
# Unless the submission is chained: with --dependency the upstream job is still writing the
# checkpoints, so absence now says nothing. Warn and let the job resolve them when it runs --
# run_lora_sweep.sh passes the path straight to the edit script, which fails loudly per task.
deferred=0
for arg in "$@"; do
  case "$arg" in --dependency=*) deferred=1 ;; esac
done
missing=0
for ckpt in "${LORA_CHECKPOINTS[@]}"; do
  [ -n "$ckpt" ] || continue   # the paired no-LoRA arm
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
if [ "$missing" -ne 0 ] && [ "$deferred" -eq 1 ]; then
  echo "  ^ not on disk yet, but this submission is chained -- the upstream job should write" >&2
  echo "    them before it runs. Submitting anyway; a still-missing one fails that task only." >&2
  missing=0
fi
[ "$missing" -eq 0 ] || exit 1

for i in "${!CONFIGS[@]}"; do
  # Rows carry an optional 4th field, steps. Legacy grids emit three and fall back to
  # LORA_STEPS; their lora_sweep_run_name ignores the extra positional, so names are unchanged.
  IFS='|' read -r ckpt tstart cfg steps <<< "${CONFIGS[$i]}"
  steps="${steps:-${LORA_STEPS:-100}}"
  printf '  %2d  tstart=%-3s cfg_tar=%-4s steps=%-4s %s\n' "$i" "$tstart" "$cfg" "$steps" \
    "$(lora_sweep_run_name "$ckpt" "$tstart" "$cfg" "$steps")"
done

REF_DIR="$(python3 -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env; print(env.medley_split_paths("'"$LORA_SPLIT"'")[1])' 2>/dev/null || true)"
if [ -n "$REF_DIR" ]; then
  echo "reference : $(find "$REF_DIR" -name 'a*.wav' 2>/dev/null | wc -l) wavs in $REF_DIR"
fi

EXPORTS=("SWEEP_CONFIGS=$SWEEP_CONFIGS")
[ -n "${SCRIPT:-}" ] && EXPORTS+=("SCRIPT=$SCRIPT")
# The job archives what it just wrote, so it needs the same subdir the eval is told about.
[ -n "${LORA_EDITS_SUBDIR:-}" ] && EXPORTS+=("LORA_EDITS_SUBDIR=$LORA_EDITS_SUBDIR")
[ -n "${PROBE:-}" ] && EXPORTS+=("PROBE=$PROBE")
[ -n "${METHOD:-}" ] && EXPORTS+=("METHOD=$METHOD")
# The grid file reads these, so the job must see the same values this preview used.
[ -n "${SPLIT:-}" ] && EXPORTS+=("SPLIT=$SPLIT")
# Spaces do not survive sbatch's --export list (the variable arrives truncated or not at all),
# so multi-value CFG_TARS travels colon-separated; the grid files split on colons and spaces.
[ -n "${CFG_TARS:-}" ] && EXPORTS+=("CFG_TARS=${CFG_TARS// /:}")
if [ -n "${SKIP_EXISTING:-}" ]; then
  echo "skip_existing: on (resuming)"
  EXPORTS+=("SKIP_EXISTING=$SKIP_EXISTING")
fi
EXPORT_ARGS=(--export="ALL,$(IFS=,; echo "${EXPORTS[*]}")")

sbatch \
  "${EXPORT_ARGS[@]}" \
  --account="$HPC_PWR_ACCOUNT" \
  --partition="$HPC_PWR_PARTITION" \
  "$@" \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_lora_sweep.sh
