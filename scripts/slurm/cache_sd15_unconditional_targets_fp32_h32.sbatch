#!/bin/bash
#SBATCH --job-name=sd15-uncond-cache
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=0-8%1
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve project storage from $REPO_DIR/checkpoints" >&2; exit 2; }
CFG_SCALE=${1:?Pass the trajectory CFG scale: 1.0 or 2.5}
case "$CFG_SCALE" in
  1|1.0) DATA_ROOT=data/processed/sd15_trajectories_stacked_fp32/all ;;
  2.5) DATA_ROOT=data/processed/sd15_trajectories_stacked_fp32_cfg2_5/all ;;
  *) echo "Unsupported CFG scale: $CFG_SCALE (expected 1.0 or 2.5)" >&2; exit 2 ;;
esac
STARTS=(0 3425 6850 10275 13700 17125 20550 23975 27400)
COUNTS=(3425 3425 3425 3425 3425 3425 3425 3425 3041)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
START_INDEX=${STARTS[$TASK_ID]:-}
NUM_SAMPLES=${COUNTS[$TASK_ID]:-}
[[ -n "$START_INDEX" && -n "$NUM_SAMPLES" ]] || { echo "Invalid array task: $TASK_ID" >&2; exit 2; }

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-uncond-cache-cfg${CFG_SCALE//./_}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"

cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$REPO_DIR/logs/slurm"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

echo "Caching unconditional targets for full-FP32 CFG $CFG_SCALE samples [$START_INDEX:$((START_INDEX + NUM_SAMPLES)))"
python3 -m diff_inversion.data.precompute_training_cache \
  root_dirs="[$DATA_ROOT]" \
  target_branches=unconditional \
  require_existing_conditional_target=true \
  batch_size=32 \
  model.torch_dtype=float32 \
  model.variant=null \
  start_index="$START_INDEX" \
  num_samples="$NUM_SAMPLES"
