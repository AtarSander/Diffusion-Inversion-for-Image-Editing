#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/pr2/projects/plgrid/plggzzsn2026/diffusion_inversion/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-/net/tscratch/people/plgmichalsadowski/conda_envs/diff-inversion/bin/python}
CONFIG_NAME=${CONFIG_NAME:-sample_gather}

OUTPUT_DIR=${OUTPUT_DIR:-data/processed/sdxl_train_100}
PROMPTS_JSONL=${PROMPTS_JSONL:-data/processed/recap_coco/train.jsonl}
CHUNK_SIZE=${CHUNK_SIZE:-100}
BASE_INDEX=${BASE_INDEX:-0}
SEED=${SEED:-1234}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
OVERWRITE=${OVERWRITE:-false}

ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_INDEX=$((BASE_INDEX + ARRAY_TASK_ID * CHUNK_SIZE))

cd "$REPO_DIR"

module load Miniconda3/23.3.1-0

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgmichalsadowski/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgmichalsadowski/wandb-cache}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"

echo "Repo: $REPO_DIR"
echo "Python: $PYTHON"
echo "Hydra config: $CONFIG_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Prompts JSONL: $PROMPTS_JSONL"
echo "Overwrite: $OVERWRITE"
echo "Array task id: $ARRAY_TASK_ID"
echo "Generating records [$START_INDEX:$((START_INDEX + CHUNK_SIZE)))"

"$PYTHON" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

"$PYTHON" -m diff_inversion.data.generate_sdxl_samples \
  --config-name "$CONFIG_NAME" \
  output_dir="$OUTPUT_DIR" \
  data.prompts_jsonl="$PROMPTS_JSONL" \
  num_samples="$CHUNK_SIZE" \
  start_index="$START_INDEX" \
  seed="$SEED" \
  overwrite="$OVERWRITE" \
  model.guidance_scale="$GUIDANCE_SCALE" \
  model.num_inference_steps="$NUM_INFERENCE_STEPS"
