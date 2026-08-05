#!/bin/bash
#SBATCH --job-name=sd15-traj-fp32
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=0-8%1
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?${SLURM_SUBMIT_DIR} is required}
BASE=$REPO_DIR
PROMPTS_JSONL=$BASE/data/processed/recap_coco/all_prompts.jsonl
OUTPUT_DIR=$BASE/data/processed/sd15_trajectories_stacked_fp32/all
STARTS=(0 3425 6850 10275 13700 17125 20550 23975 27400)
COUNTS=(3425 3425 3425 3425 3425 3425 3425 3425 3041)
TASK_ID=${SLURM_ARRAY_TASK_ID:?${SLURM_ARRAY_TASK_ID} is required}
START_INDEX=${STARTS[$TASK_ID]:-}
NUM_SAMPLES=${COUNTS[$TASK_ID]:-}
[[ -n "$START_INDEX" && -n "$NUM_SAMPLES" ]] || { echo "Invalid array task: $TASK_ID"; exit 2; }

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-traj-fp32-${SLURM_ARRAY_JOB_ID}-$TASK_ID
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export HF_HOME=$BASE/cache/h86/sd15_trajectories_fp32/huggingface
export UV_CACHE_DIR=$REPO_DIR/.uv-cache

cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$OUTPUT_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

echo "Generating $NUM_SAMPLES full-FP32 trajectories: [$START_INDEX:$((START_INDEX + NUM_SAMPLES)))"
python3 -m diff_inversion.data.generate_sdxl_samples --config-name sample_gather_sd15 \
  output_dir="$OUTPUT_DIR" \
  data.prompts_jsonl="$PROMPTS_JSONL" \
  start_index="$START_INDEX" \
  num_samples="$NUM_SAMPLES" \
  model.torch_dtype=float32 \
  model.variant=null
