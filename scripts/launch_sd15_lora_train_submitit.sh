#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
UV=${UV:-uv}

DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggzzsn2026/plgatarsander/data/processed/sd15_trajectories_stacked}
TRAIN_ROOT=${TRAIN_ROOT:-$DATA_ROOT/train}
VAL_ROOT=${VAL_ROOT:-$DATA_ROOT/val}
RUN_NAME=${RUN_NAME:-sd15-inversion-lora-r16-lr5e-5-cosine-fp16}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/net/pr2/projects/plgrid/plggzzsn2026/plgatarsander/checkpoints/sd15_inversion_lora_r16_lr5e-5_cosine_fp16}

cd "$REPO_DIR"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/net/tscratch/people/plgatarsander/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO_DIR/.venv}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$MPLCONFIGDIR" "$CHECKPOINT_DIR"

if [[ -x "$PYTHON" ]]; then
  exec "$PYTHON" -m diff_inversion.modeling.train \
    --config-name train_sd15_submitit \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=sd15_lora_train \
    data.root_dir="$TRAIN_ROOT" \
    data.val_root_dir="$VAL_ROOT" \
    run_name="$RUN_NAME" \
    checkpoint_dir="$CHECKPOINT_DIR" \
    "$@"
fi

exec "$UV" run --frozen python -m diff_inversion.modeling.train \
  --config-name train_sd15_submitit \
  --multirun \
  'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.name=sd15_lora_train \
  data.root_dir="$TRAIN_ROOT" \
  data.val_root_dir="$VAL_ROOT" \
  run_name="$RUN_NAME" \
  checkpoint_dir="$CHECKPOINT_DIR" \
  "$@"

