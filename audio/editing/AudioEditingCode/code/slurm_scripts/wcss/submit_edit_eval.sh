#!/bin/bash
# ABOUTME: One-command submission of a chained edit sweep + eval pair from a grid file, with
# ABOUTME: robust job-id capture and retries around WCSS's transient sbatch socket timeouts.
#
# Usage (from anywhere; KEY=VALUE args in any order):
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_edit_eval.sh \
#     GRID=sao_nfe_matched_configs.sh ARRAY=0-23 METHOD=odeinv CFG_TARS="7.0 10.5 14.0"
#
#   GRID    grid file basename in this directory (required)
#   ARRAY   sbatch array spec for both the edit and the eval job (required)
#   DEP     optional upstream job id; the edit waits on it with afterok
#   EVAL_ONLY=1  skip the edit job and submit only the eval for the grid (no dependency unless DEP)
#   METHOD/CFG_TARS/SPLIT/PROBE  forwarded to the grid file as usual
#   SCRIPT  edit entrypoint (default edit_stableaudio_medleydb.py)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$AUDIO_ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) echo "ERROR: arguments must be KEY=VALUE, got '$arg'" >&2; exit 2 ;;
  esac
done
: "${GRID:?usage: submit_edit_eval.sh GRID=<grid file> ARRAY=<spec> [KEY=VALUE...]}"
: "${ARRAY:?usage: submit_edit_eval.sh GRID=<grid file> ARRAY=<spec> [KEY=VALUE...]}"
[ -f "$HERE/$GRID" ] || { echo "ERROR: no grid file $HERE/$GRID" >&2; exit 2; }
export SWEEP_CONFIGS="editing/AudioEditingCode/code/slurm_scripts/wcss/$GRID"
export SCRIPT="${SCRIPT:-edit_stableaudio_medleydb.py}"

# Transient "Socket timed out on send/recv operation" failures are routine on this controller;
# nothing is submitted when sbatch fails, so retrying the whole submit script is safe.
submit() { # submit <label> <command...> -> prints the job id
  local label="$1" attempt out id; shift
  for attempt in 1 2 3; do
    if out="$("$@" 2>&1)"; then
      id="$(grep -oE 'Submitted batch job [0-9]+' <<< "$out" | awk 'END{print $4}')"
      if [ -n "$id" ]; then echo "$out" | grep -vE '^Submitted batch job' >&2
        echo "$label: job $id" >&2; echo "$id"; return 0; fi
    fi
    echo "$out" >&2
    echo "WARNING: $label submission attempt $attempt failed; retrying in 30s" >&2
    sleep 30
  done
  echo "ERROR: $label submission failed after 3 attempts" >&2
  return 1
}

EVAL_DEP="${DEP:-}"
if [ "${EVAL_ONLY:-0}" != "1" ]; then
  EDIT_FLAGS=(--array="$ARRAY")
  [ -n "${DEP:-}" ] && EDIT_FLAGS+=(--dependency=afterok:"$DEP")
  EDIT_ID="$(submit edit bash "$HERE/submit_lora_sweep.sh" "${EDIT_FLAGS[@]}")"
  EVAL_DEP="$EDIT_ID"
fi

EVAL_FLAGS=(--array="$ARRAY")
[ -n "$EVAL_DEP" ] && EVAL_FLAGS+=(--dependency=afterok:"$EVAL_DEP")
EVAL_ID="$(ARM=lora submit eval bash "$HERE/submit_eval.sh" "${EVAL_FLAGS[@]}")"

echo "submitted: edit=${EDIT_ID:-skipped} eval=$EVAL_ID (grid=$GRID array=$ARRAY)"
