#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON_CMD=${PYTHON_CMD:-"uv run python"}
DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggzzsn2026/plgatarsander/data/processed/sd15_trajectories_stacked}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-$DATA_ROOT/train}
VAL_OUTPUT_DIR=${VAL_OUTPUT_DIR:-$DATA_ROOT/val}
TRAIN_PROMPTS_JSONL=${TRAIN_PROMPTS_JSONL:-/net/tscratch/people/plgatarsander/ZZSN_data/processed/recap_coco/train.jsonl}
VAL_PROMPTS_JSONL=${VAL_PROMPTS_JSONL:-/net/tscratch/people/plgatarsander/ZZSN_data/processed/recap_coco/val.jsonl}

GENERATE_TRAIN=${GENERATE_TRAIN:-true}
GENERATE_VAL=${GENERATE_VAL:-true}
TRAIN_LORA=${TRAIN_LORA:-true}

TRAIN_NUM_SAMPLES=${TRAIN_NUM_SAMPLES:-27396}
VAL_NUM_SAMPLES=${VAL_NUM_SAMPLES:-1522}
TRAIN_START_INDEX=${TRAIN_START_INDEX:-0}
VAL_START_INDEX=${VAL_START_INDEX:-0}
SEED=${SEED:-1234}
OVERWRITE=${OVERWRITE:-false}

RUN_NAME=${RUN_NAME:-sd15-inversion-lora-r16-lr5e-5-cosine-fp16}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/net/pr2/projects/plgrid/plggzzsn2026/plgatarsander/checkpoints/sd15_inversion_lora_r16_lr5e-5_cosine_fp16}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-171225}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-diff-inversion}

cd "$REPO_DIR"
read -r -a PYTHON_ARR <<< "$PYTHON_CMD"

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export WANDB_DIR=${WANDB_DIR:-/net/tscratch/people/plgatarsander/wandb}
mkdir -p "$TRAIN_OUTPUT_DIR" "$VAL_OUTPUT_DIR" "$CHECKPOINT_DIR" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR"

run_python() {
  "${PYTHON_ARR[@]}" "$@"
}

echo "Repo: $REPO_DIR"
echo "Python command: $PYTHON_CMD"
echo "Data root: $DATA_ROOT"
echo "Train output: $TRAIN_OUTPUT_DIR"
echo "Val output: $VAL_OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"

run_python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

if [[ "$GENERATE_TRAIN" == "true" ]]; then
  run_python -m diff_inversion.data.generate_sdxl_samples     --config-name sample_gather_sd15     output_dir="$TRAIN_OUTPUT_DIR"     data.prompts_jsonl="$TRAIN_PROMPTS_JSONL"     num_samples="$TRAIN_NUM_SAMPLES"     start_index="$TRAIN_START_INDEX"     seed="$SEED"     overwrite="$OVERWRITE"
fi

if [[ "$GENERATE_VAL" == "true" ]]; then
  run_python -m diff_inversion.data.generate_sdxl_samples     --config-name sample_gather_sd15     output_dir="$VAL_OUTPUT_DIR"     data.prompts_jsonl="$VAL_PROMPTS_JSONL"     num_samples="$VAL_NUM_SAMPLES"     start_index="$VAL_START_INDEX"     seed="$SEED"     overwrite="$OVERWRITE"
fi

if [[ "$TRAIN_LORA" == "true" ]]; then
  run_python -m diff_inversion.modeling.train     --config-name train_sd15     data.root_dir="$TRAIN_OUTPUT_DIR"     data.val_root_dir="$VAL_OUTPUT_DIR"     data.batch_size="$BATCH_SIZE"     gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"     max_train_steps="$MAX_TRAIN_STEPS"     run_name="$RUN_NAME"     checkpoint_dir="$CHECKPOINT_DIR"     wandb.mode="$WANDB_MODE"     wandb.project="$WANDB_PROJECT"
fi
