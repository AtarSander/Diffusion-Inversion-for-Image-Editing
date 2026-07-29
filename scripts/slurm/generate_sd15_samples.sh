#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-.venv/bin/python}
CONFIG_NAME=${CONFIG_NAME:-sample_gather_sd15}

DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/processed/sd15_trajectories_stacked}
SPLIT=${SPLIT:-train}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/$SPLIT}
PROMPTS_JSONL=${PROMPTS_JSONL:-/net/tscratch/people/plgatarsander/ZZSN_data/processed/recap_coco/${SPLIT}.jsonl}
CHUNK_SIZE=${CHUNK_SIZE:-100}
BASE_INDEX=${BASE_INDEX:-0}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-}
SEED=${SEED:-1234}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
OVERWRITE=${OVERWRITE:-false}

ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_INDEX=$((BASE_INDEX + ARRAY_TASK_ID * CHUNK_SIZE))
SAMPLES_THIS_TASK=$CHUNK_SIZE

if [[ -n "$TOTAL_SAMPLES" ]]; then
  if (( START_INDEX >= TOTAL_SAMPLES )); then
    echo "Skipping empty task: START_INDEX=$START_INDEX TOTAL_SAMPLES=$TOTAL_SAMPLES"
    exit 0
  fi

  REMAINING=$((TOTAL_SAMPLES - START_INDEX))
  if (( REMAINING < CHUNK_SIZE )); then
    SAMPLES_THIS_TASK=$REMAINING
  fi
fi

cd "$REPO_DIR"

module load Miniconda3/23.3.1-0

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgatarsander/wandb-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR" "$OUTPUT_DIR"

echo "Repo: $REPO_DIR"
echo "Python: $PYTHON"
echo "Hydra config: $CONFIG_NAME"
echo "Split: $SPLIT"
echo "Output dir: $OUTPUT_DIR"
echo "Prompts JSONL: $PROMPTS_JSONL"
echo "Overwrite: $OVERWRITE"
echo "Array task id: $ARRAY_TASK_ID"
echo "Generating records [$START_INDEX:$((START_INDEX + SAMPLES_THIS_TASK)))"

"$PYTHON" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

"$PYTHON" -m diff_inversion.data.generate_sdxl_samples \
  --config-name "$CONFIG_NAME" \
  output_dir="$OUTPUT_DIR" \
  data.prompts_jsonl="$PROMPTS_JSONL" \
  num_samples="$SAMPLES_THIS_TASK" \
  start_index="$START_INDEX" \
  seed="$SEED" \
  overwrite="$OVERWRITE" \
  model.guidance_scale="$GUIDANCE_SCALE" \
  model.num_inference_steps="$NUM_INFERENCE_STEPS"
