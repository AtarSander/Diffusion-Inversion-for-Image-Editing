#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
DATA_ROOT=${DATA_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/processed/sd15_trajectories_stacked}
SPLIT=${SPLIT:-train}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/$SPLIT}
PROMPTS_JSONL=${PROMPTS_JSONL:-/net/tscratch/people/plgatarsander/ZZSN_data/processed/recap_coco/${SPLIT}.jsonl}
CHUNK_SIZE=${CHUNK_SIZE:-3425}
BASE_INDEX=${BASE_INDEX:-0}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-}
ARRAY_PARALLELISM=${ARRAY_PARALLELISM:-8}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-$REPO_DIR/scripts/slurm/generate_sd15_samples.sh}

ACCOUNT=${ACCOUNT:-plgzzsn2026-gpu-a100}
PARTITION=${PARTITION:-plgrid-gpu-a100}
TIME=${TIME:-13:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
MEM=${MEM:-128G}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
JOB_NAME=${JOB_NAME:-sd15_sample_gather_${SPLIT}}
LOG_DIR=${LOG_DIR:-$REPO_DIR/slurm_logs/sd15_sample_gather}

SEED=${SEED:-1234}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
OVERWRITE=${OVERWRITE:-false}

cd "$REPO_DIR"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "Prompt JSONL not found: $PROMPTS_JSONL" >&2
  exit 1
fi

if [[ -z "$TOTAL_SAMPLES" ]]; then
  TOTAL_SAMPLES=$(wc -l < "$PROMPTS_JSONL")
fi

if (( TOTAL_SAMPLES <= BASE_INDEX )); then
  echo "Nothing to submit: TOTAL_SAMPLES=$TOTAL_SAMPLES BASE_INDEX=$BASE_INDEX" >&2
  exit 1
fi

NUM_TASKS=$(( (TOTAL_SAMPLES - BASE_INDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_LAST=$(( NUM_TASKS - 1 ))

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Submitting SD15 sample gathering array"
echo "Repo: $REPO_DIR"
echo "Split: $SPLIT"
echo "Output dir: $OUTPUT_DIR"
echo "Prompts JSONL: $PROMPTS_JSONL"
echo "Total samples: $TOTAL_SAMPLES"
echo "Base index: $BASE_INDEX"
echo "Chunk size: $CHUNK_SIZE"
echo "Array: 0-${ARRAY_LAST}%${ARRAY_PARALLELISM}"

export REPO_DIR OUTPUT_DIR PROMPTS_JSONL CHUNK_SIZE BASE_INDEX TOTAL_SAMPLES
export SEED GUIDANCE_SCALE NUM_INFERENCE_STEPS OVERWRITE

sbatch \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gpus-per-node="$GPUS_PER_NODE" \
  --array="0-${ARRAY_LAST}%${ARRAY_PARALLELISM}" \
  --output="$LOG_DIR/%x_%A_%a.out" \
  --error="$LOG_DIR/%x_%A_%a.err" \
  --export=ALL \
  "$SBATCH_SCRIPT" \
  "$@"
