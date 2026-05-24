#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/pr2/projects/plgrid/plggzzsn2026/diffusion_inversion/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-/net/tscratch/people/plgmichalsadowski/conda_envs/diff-inversion/bin/python}

cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module load Miniconda3/23.3.1-0
fi

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgmichalsadowski/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgmichalsadowski/wandb-cache}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"

exec "$PYTHON" diff_inversion/eval/invert_sdxl.py \
  --config-name eval_invert_sdxl_submitit \
  --multirun \
  'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.name=sdxl_eval_invert \
  "$@"
