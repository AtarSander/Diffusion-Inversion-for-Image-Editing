#!/bin/bash -l
# ABOUTME: PWR/WCSS array job editing the 115-row MedleyMD hparam split with AudioLDM2 across the
# ABOUTME: method x tstart x cfg_tar grid, one array task per config, for the tradeoff curves.
#SBATCH --job-name=medleymd-hparam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-47
#SBATCH --output=outputs/logs/slurm/hparam-%A_%a.out
#SBATCH --error=outputs/logs/slurm/hparam-%A_%a.err
#
# Submit from audio/:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_hparam_sweep.sh
#   SKIP_EXISTING=1 bash .../submit_hparam_sweep.sh --array=0-15   # resume, ddpm only
#
# One task = one config over all 115 rows, no sharding: the slowest config (ddpm at tstart=200,
# ~130 s per edit) is about 4 h, and skip_existing makes a task killed by the limit resumable.
# Prerequisites:
#   python -m editing.build_lower_bound --splits hparam   # 115-row paired reference
#   the same .venv and HF cache the baseline array uses

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

# SLURM spools only the batch script, so the helper has to be sourced from the submit directory.
source "editing/AudioEditingCode/code/slurm_scripts/wcss/hparam_sweep_configs.sh" || exit 1
mapfile -t CONFIGS < <(sweep_configs)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
if [ "$TASK_ID" -ge "${#CONFIGS[@]}" ]; then
  echo "array index $TASK_ID exceeds ${#CONFIGS[@]} configs" >&2
  exit 2
fi

IFS='|' read -r MODE TSTART CFG_TAR <<< "${CONFIGS[$TASK_ID]}"
RUN_NAME="$(sweep_run_name "$MODE" "$TSTART" "$CFG_TAR")"
: "${RUN_NAME:?run name resolved empty; edits would land in the parent directory}"
echo "task=$TASK_ID mode=$MODE tstart=$TSTART cfg_tar=$CFG_TAR run=$RUN_NAME node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd editing/AudioEditingCode/code

ok=0
for attempt in 1 2 3; do
  python edit_audioldm_medleydb.py \
    --mode "$MODE" \
    --num_diffusion_steps "$SWEEP_STEPS" \
    --cfg_src "$SWEEP_CFG_SRC" \
    --cfg_tar "$CFG_TAR" \
    --tstart "$TSTART" \
    --seed 42 \
    --split "$SWEEP_SPLIT" \
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
    --run_dir "$(python -c 'import sys; sys.path.insert(0, "editing/AudioEditingCode/code"); import env, pathlib; print(pathlib.Path(env.PATH_EDIT_OUTPUTS) / "medleymd" / sys.argv[1] / sys.argv[2])' "audioldm2_$MODE" "$RUN_NAME")" \
    --remove True --overwrite True || echo "WARNING: archiving failed for $RUN_NAME; audios/ kept" >&2
fi

echo "done: $RUN_NAME"
