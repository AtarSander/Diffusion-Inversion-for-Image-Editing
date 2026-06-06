#!/bin/bash
set -euo pipefail

if [[ -x /net/people/plgrid/plgmichalsadowski/.local/bin/uv ]]; then
  exec /net/people/plgrid/plgmichalsadowski/.local/bin/uv run python diff_inversion/modeling/train.py \
    --config-name train_submitit \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=sdxl_lora_train \
    "$@"
fi

cd "$REPO_DIR"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/net/tscratch/people/plgatarsander/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/net/tscratch/people/plgatarsander/venv}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR"

exec "$UV" run --frozen python diff_inversion/modeling/train.py \
  --config-name train_submitit \
  --multirun \
  'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.name=sdxl_lora_train \
  "$@"
