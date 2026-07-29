#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-.venv/bin/python}
CONFIG_NAME=${CONFIG_NAME:-train_sd15}

DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/processed/sd15_trajectories_stacked}
TRAIN_ROOT=${TRAIN_ROOT:-$DATA_ROOT/train}
VAL_ROOT=${VAL_ROOT:-$DATA_ROOT/val}
RUN_NAME=${RUN_NAME:-sd15-inversion-lora-r16-lr5e-5-cosine-bf16}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/checkpoints/sd15_inversion_lora_r16_lr5e-5_cosine_bf16}

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-171225}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LOG_EVERY_STEPS=${LOG_EVERY_STEPS:-50}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-1000}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-5000}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-128}
VALIDATION_PREVIEW_ENABLED=${VALIDATION_PREVIEW_ENABLED:-false}
VALIDATION_PREVIEW_EVERY_STEPS=${VALIDATION_PREVIEW_EVERY_STEPS:-$EVAL_EVERY_STEPS}
VALIDATION_PREVIEW_NUM_SAMPLES=${VALIDATION_PREVIEW_NUM_SAMPLES:-2}
VALIDATION_PREVIEW_LOG_PREDICTED_NOISE=${VALIDATION_PREVIEW_LOG_PREDICTED_NOISE:-true}
VALIDATION_PREVIEW_LOG_INVERTED_NOISE=${VALIDATION_PREVIEW_LOG_INVERTED_NOISE:-true}
VALIDATION_PREVIEW_CALCULATE_LPIPS=${VALIDATION_PREVIEW_CALCULATE_LPIPS:-false}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-diff-inversion}

cd "$REPO_DIR"

module load Miniconda3/23.3.1-0

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$MPLCONFIGDIR" "$CHECKPOINT_DIR"

echo "Repo: $REPO_DIR"
echo "Python: $PYTHON"
echo "Hydra config: $CONFIG_NAME"
echo "Train root: $TRAIN_ROOT"
echo "Val root: $VAL_ROOT"
echo "Run name: $RUN_NAME"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Max train steps: $MAX_TRAIN_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

"$PYTHON" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

"$PYTHON" -m diff_inversion.modeling.train   --config-name "$CONFIG_NAME"   data.root_dir="$TRAIN_ROOT"   data.val_root_dir="$VAL_ROOT"   data.batch_size="$BATCH_SIZE"   wandb.mode="$WANDB_MODE"   wandb.project="$WANDB_PROJECT"   run_name="$RUN_NAME"   checkpoint_dir="$CHECKPOINT_DIR"   learning_rate="$LEARNING_RATE"   max_train_steps="$MAX_TRAIN_STEPS"   gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"   log_every_steps="$LOG_EVERY_STEPS"   eval_every_steps="$EVAL_EVERY_STEPS"   save_every_steps="$SAVE_EVERY_STEPS"   max_val_batches="$MAX_VAL_BATCHES"   validation_preview.enabled="$VALIDATION_PREVIEW_ENABLED"   validation_preview.every_steps="$VALIDATION_PREVIEW_EVERY_STEPS"   validation_preview.num_samples="$VALIDATION_PREVIEW_NUM_SAMPLES"   validation_preview.log_predicted_noise="$VALIDATION_PREVIEW_LOG_PREDICTED_NOISE"   validation_preview.log_inverted_noise="$VALIDATION_PREVIEW_LOG_INVERTED_NOISE"   validation_preview.calculate_lpips="$VALIDATION_PREVIEW_CALCULATE_LPIPS"
