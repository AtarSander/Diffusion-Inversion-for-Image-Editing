#!/bin/bash -l
# ABOUTME: PWR/WCSS array job editing the 115-row hparam split with DDIM inversion under a trained
# ABOUTME: inversion LoRA, one task per (checkpoint, tstart, cfg_tar), for the tradeoff curves.
#SBATCH --job-name=medleymd-lorasweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-63
#SBATCH --output=outputs/logs/slurm/lorasweep-%A_%a.out
#SBATCH --error=outputs/logs/slurm/lorasweep-%A_%a.err
#
# Submit from audio/:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_lora_sweep.sh
#   bash .../submit_lora_sweep.sh --array=0-7    # one checkpoint only
#   bash .../submit_lora_sweep.sh --array=48-63 # the timestep-embedding pair only
#
# The no-LoRA twin of every config is already scored by run_hparam_sweep.sh, so each of these
# runs has a paired reference at identical settings over the same 115 rows.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

# SLURM spools only the batch script, so the helper is sourced from the submit directory.
source "editing/AudioEditingCode/code/slurm_scripts/wcss/lora_sweep_configs.sh" || exit 1
mapfile -t CONFIGS < <(lora_sweep_configs)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
if [ "$TASK_ID" -ge "${#CONFIGS[@]}" ]; then
  echo "array index $TASK_ID exceeds ${#CONFIGS[@]} configs" >&2
  exit 2
fi

IFS='|' read -r CKPT TSTART CFG_TAR <<< "${CONFIGS[$TASK_ID]}"
LORA_PATH="${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}/$CKPT"
RUN_NAME="$(lora_sweep_run_name "$CKPT" "$TSTART" "$CFG_TAR")"
: "${RUN_NAME:?run name resolved empty; edits would land in the parent directory}"

if [ ! -f "$LORA_PATH" ] || [ ! -f "${LORA_PATH%.pt}.json" ]; then
  echo "ERROR: missing checkpoint or sidecar for $LORA_PATH" >&2
  exit 3
fi

echo "task=$TASK_ID ckpt=$CKPT tstart=$TSTART cfg_tar=$CFG_TAR run=$RUN_NAME node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd editing/AudioEditingCode/code

ok=0
for attempt in 1 2 3; do
  python edit_audioldm_medleydb.py \
    --mode ddim \
    --num_diffusion_steps "$LORA_STEPS" \
    --cfg_src "$LORA_CFG_SRC" \
    --cfg_tar "$CFG_TAR" \
    --tstart "$TSTART" \
    --seed 42 \
    --split "$LORA_SPLIT" \
    --lora_path "$LORA_PATH" \
    --run_name "$RUN_NAME" ${SKIP_EXISTING:+--skip_existing True} && { ok=1; break; }
  echo "attempt $attempt failed for $RUN_NAME; retrying in 120s..." >&2
  sleep 120
done
if [ "$ok" -ne 1 ]; then
  echo "FAILED after 3 attempts: $RUN_NAME" >&2
  exit 1
fi

# Pack the per-example wavs into one tar: a finished run then costs a handful of inodes instead
# of 115, and the eval unpacks it to node-local scratch. Only after a successful edit, and
# archive_run verifies every file is in the archive before it removes anything.
if [ "${ARCHIVE_RUNS:-1}" = "1" ]; then
  cd "${SLURM_SUBMIT_DIR:-$PWD}"
  PYTHONPATH="$PWD:${PYTHONPATH:-}" python -m editing.archive_run pack \
    --run_dir "$(python -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env, pathlib; print(pathlib.Path(env.PATH_EDIT_OUTPUTS) / "medleymd" / sys.argv[1] / sys.argv[2])' "audioldm2_ddim" "$RUN_NAME")" \
    --remove True --overwrite True || echo "WARNING: archiving failed for $RUN_NAME; audios/ kept" >&2
fi

echo "done: $RUN_NAME"
