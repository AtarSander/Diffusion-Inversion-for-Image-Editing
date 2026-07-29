#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
UV=${UV:-uv}

DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/processed/sd15_trajectories_stacked}
TRAIN_ROOT=${TRAIN_ROOT:-$DATA_ROOT/train}
VAL_ROOT=${VAL_ROOT:-$DATA_ROOT/val}
LOG_ROOT=${LOG_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/logs}
SWEEP_DIR=${SWEEP_DIR:-$LOG_ROOT/sd15_lora_bf16_lr5e-6_rank_sweep}
TRAIN_CONFIGS=${TRAIN_CONFIGS:-sd15_lora_lr5em6_r8,sd15_lora_lr5em6,sd15_lora_lr5em6_r32}

cd "$REPO_DIR"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/net/tscratch/people/plgatarsander/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO_DIR/.venv}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$MPLCONFIGDIR" "$SWEEP_DIR"

run_train() {
  if [[ -x "$PYTHON" ]]; then
    "$PYTHON" -m diff_inversion.modeling.train "$@"
    return
  fi

  "$UV" run --frozen python -m diff_inversion.modeling.train "$@"
}

echo "Submitting SD1.5 LoRA bf16 lr=5e-6 rank sweep: $TRAIN_CONFIGS"
echo "Train root: $TRAIN_ROOT"
echo "Val root: $VAL_ROOT"
echo "Logs: $SWEEP_DIR"

run_train \
  --config-name train_sd15_submitit \
  --multirun \
  "hydra.sweep.dir=$SWEEP_DIR/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
  hydra.job.name=sd15_lora_lowlr_rank_sweep \
  "train=$TRAIN_CONFIGS" \
  data.root_dir="$TRAIN_ROOT" \
  data.val_root_dir="$VAL_ROOT" \
  "$@"
