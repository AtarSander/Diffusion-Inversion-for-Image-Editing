#!/usr/bin/env bash
# Generate PIE-Bench-prompt SDXL trajectories and evaluate DDIM/r8/r16/r32 on them.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME_GENERATE=${SLURM_TIME_GENERATE:-12:00:00}
SLURM_TIME_EVAL=${SLURM_TIME_EVAL:-24:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-48G}
CHUNK_SIZE=${CHUNK_SIZE:-25}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_pie_bench_ood_inversion_quality}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/checkpoints}
TRAJECTORY_ROOT=${TRAJECTORY_ROOT:-$REPO_DIR/data/processed/sdxl_trajectories_pie_bench_ood/all}
REPORT_ROOT=${REPORT_ROOT:-$REPO_DIR/reports/eval/sdxl_pie_bench_ood}
TOTAL_PROMPTS=${TOTAL_PROMPTS:-700}

if [[ ${1:-} != --generate-worker && ${1:-} != --evaluate-worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  "$PYTHON" -m diff_inversion.data.prepare_pie_bench_ood_prompts
  TASKS=$(((TOTAL_PROMPTS + CHUNK_SIZE - 1) / CHUNK_SIZE))
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  GENERATE_JOB=$(sbatch --parsable --job-name=sdxl-pie-ood-gen --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --gres="$SLURM_GRES" \
    --time="$SLURM_TIME_GENERATE" --array="0-$((TASKS - 1))" \
    --output="$RUN_LOG_DIR/generate_%A_%a.out" --error="$RUN_LOG_DIR/generate_%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",TRAJECTORY_ROOT="$TRAJECTORY_ROOT",CHUNK_SIZE="$CHUNK_SIZE",TOTAL_PROMPTS="$TOTAL_PROMPTS" \
    "$0" --generate-worker)
  EVALUATE_JOB=$(sbatch --parsable --dependency="afterok:${GENERATE_JOB}" --job-name=sdxl-pie-ood-eval --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --gres="$SLURM_GRES" \
    --time="$SLURM_TIME_EVAL" --array=0-3 \
    --output="$RUN_LOG_DIR/evaluate_%A_%a.out" --error="$RUN_LOG_DIR/evaluate_%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",TRAJECTORY_ROOT="$TRAJECTORY_ROOT",REPORT_ROOT="$REPORT_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
    "$0" --evaluate-worker)
  echo "Trajectory array: $GENERATE_JOB"
  echo "DDIM/LoRA evaluation array (after generation): $EVALUATE_JOB"
  exit 0
fi

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-pie-ood-${SLURM_JOB_ID}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sdxl_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"
MODEL_DIR="/dev/shm/${USER:-unknown}/sdxl-shared-model/stable-diffusion-xl-base-1.0"
stage_sdxl_model "$MODEL_DIR"

if [[ $1 == --generate-worker ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  (( START_INDEX < TOTAL_PROMPTS )) || exit 0
  NUM_SAMPLES=$CHUNK_SIZE
  (( START_INDEX + NUM_SAMPLES <= TOTAL_PROMPTS )) || NUM_SAMPLES=$((TOTAL_PROMPTS - START_INDEX))
  "$PYTHON" -m diff_inversion.data.generate_sdxl_samples \
    --config-name sample_gather_sdxl_pie_bench_ood \
    model.model_id="$MODEL_DIR" output_dir="$TRAJECTORY_ROOT" start_index="$START_INDEX" num_samples="$NUM_SAMPLES"
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$SLURM_ARRAY_TASK_ID" in
  0) METHOD=ddim; RANK=0; ALPHA=0 ;;
  1) METHOD=lora_r8; RANK=8; ALPHA=4 ;;
  2) METHOD=lora_r16; RANK=16; ALPHA=8 ;;
  3) METHOD=lora_r32; RANK=32; ALPHA=16 ;;
  *) echo "Unexpected evaluation task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
WORK_ROOT="$REPO_DIR/data/processed/sdxl_trajectories_pie_bench_ood/$METHOD"
OUTPUT_DIR="$REPORT_ROOT/$METHOD"
if [[ ! -d $WORK_ROOT ]]; then
  mkdir -p "$(dirname "$WORK_ROOT")"
  cp -al "$TRAJECTORY_ROOT" "$WORK_ROOT"
fi
if [[ $METHOD == ddim ]]; then
  "$PYTHON" -m diff_inversion.eval.invert_sdxl input_dir="$WORK_ROOT" overwrite=true lora.enabled=false save_inversion_artifacts=false model_id_override="$MODEL_DIR"
else
  CHECKPOINT_PATH="$CHECKPOINT_ROOT/sdxl_inversion_lora_r${RANK}_lr5e-5_cosine_bf16/checkpoint_step_15000.pt"
  [[ -f $CHECKPOINT_PATH ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 2; }
  "$PYTHON" -m diff_inversion.eval.invert_sdxl input_dir="$WORK_ROOT" overwrite=true \
lora.r="$RANK" lora.lora_alpha="$ALPHA" lora.enabled=true lora.checkpoint_path="$CHECKPOINT_PATH" save_inversion_artifacts=false model_id_override="$MODEL_DIR"
fi
"$PYTHON" -m diff_inversion.eval.reconstruct_sdxl input_dir="$WORK_ROOT" overwrite=true model_id_override="$MODEL_DIR"
"$PYTHON" -m diff_inversion.eval.run input_dir="$WORK_ROOT" output_dir="$OUTPUT_DIR" max_samples="$TOTAL_PROMPTS" wandb.mode=disabled
