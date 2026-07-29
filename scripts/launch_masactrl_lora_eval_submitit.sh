#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
UV=${UV:-uv}
if [[ -z "${HYDRA_SWEEP_DIR:-}" ]]; then
  HYDRA_SWEEP_DIR='/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/logs/masactrl_lora_eval/${now:%Y-%m-%d}/${now:%H-%M-%S}'
fi

cd "$REPO_DIR"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/net/tscratch/people/plgatarsander/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO_DIR/.venv}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$MPLCONFIGDIR"

if [[ -x "$PYTHON" ]]; then
  exec "$PYTHON" -m diff_inversion.eval.masactrl_lora \
    --config-name eval_masactrl_lora_submitit \
    --multirun \
    "hydra.sweep.dir=$HYDRA_SWEEP_DIR" \
    hydra.job.name=masactrl_lora_eval \
    "$@"
fi

exec "$UV" run --frozen python -m diff_inversion.eval.masactrl_lora \
  --config-name eval_masactrl_lora_submitit \
  --multirun \
  "hydra.sweep.dir=$HYDRA_SWEEP_DIR" \
  hydra.job.name=masactrl_lora_eval \
  "$@"
