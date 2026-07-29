#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
DATA_PATH=${DATA_PATH:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/raw/PIE-Bench_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/direct_inversion_reproduction_outputs}
LOG_ROOT=${LOG_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/logs/direct_inversion_reproduction}
ARRAY_PARALLELISM=${ARRAY_PARALLELISM:-8}
ARRAY_TASKS=${ARRAY_TASKS:-0-7}
RERUN_EXIST_IMAGES=${RERUN_EXIST_IMAGES:-false}
P2P_DTYPE=${P2P_DTYPE:-fp16}

if [[ ${1:-} != "--worker" ]]; then
  RUN_ID=$(date +%Y-%m-%d/%H-%M-%S)
  RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
  mkdir -p "$RUN_LOG_DIR"

  exec sbatch \
    --job-name=direct_inversion_reproduction \
    --account=plgdiffusion3-gpu-a100 \
    --partition=plgrid-gpu-a100 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=128G \
    --gres=gpu:1 \
    --time=13:00:00 \
    --array="${ARRAY_TASKS}%${ARRAY_PARALLELISM}" \
    --output="$RUN_LOG_DIR/%A_%a.out" \
    --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",DATA_PATH="$DATA_PATH",OUTPUT_ROOT="$OUTPUT_ROOT",RERUN_EXIST_IMAGES="$RERUN_EXIST_IMAGES",P2P_DTYPE="$P2P_DTYPE" \
    "$0" --worker
fi

RUNNERS=(
  run_editing_p2p.py
  run_editing_p2p.py
  run_editing_masactrl.py
  run_editing_masactrl.py
  run_editing_pix2pix_zero.py
  run_editing_pix2pix_zero.py
  run_editing_pnp.py
  run_editing_pnp.py
)
METHODS=(
  ddim+p2p
  directinversion+p2p
  ddim+masactrl
  directinversion+masactrl
  ddim+pix2pix-zero
  directinversion+pix2pix-zero
  ddim+pnp
  directinversion+pnp
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
  echo "Invalid array task id: $TASK_ID" >&2
  exit 2
fi

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR" "$OUTPUT_ROOT"

RUNNER=${RUNNERS[$TASK_ID]}
METHOD=${METHODS[$TASK_ID]}
COMMAND=(
  "$PYTHON" "$RUNNER"
  --data_path "$DATA_PATH"
  --output_path "$OUTPUT_ROOT"
  --edit_category_list 0 1 2 3 4 5 6 7 8 9
  --edit_method_list "$METHOD"
)
if [[ $RERUN_EXIST_IMAGES == true ]]; then
  COMMAND+=(--rerun_exist_images)
fi

echo "Array task $TASK_ID: $METHOD via $RUNNER"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

cd "$REPO_DIR/pnp_inversion"
exec "${COMMAND[@]}"
