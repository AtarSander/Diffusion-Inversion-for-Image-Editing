#!/usr/bin/env bash
# Evaluate the SDXL r16 conditional-cycle LoRA on ID ReCap-COCO heldout and OOD PIE-Bench prompts.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-48:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-48G}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_conditional_cycle_r16_eval}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/checkpoints}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/sdxl_inversion_lora_r16_conditional_cycle_lr5e-5_cosine_bf16/checkpoint_step_15000.pt}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  [[ -f $CHECKPOINT_PATH ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  JOB_ID=$(sbatch --parsable --job-name=sdxl-cycle-r16-eval --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --gres="$SLURM_GRES" \
    --time="$SLURM_TIME" --array=0-1 \
    --output="$RUN_LOG_DIR/eval_%A_%a.out" --error="$RUN_LOG_DIR/eval_%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    "$0" --worker)
  echo "Submitted ID/OOD cycle-loss evaluation array: $JOB_ID"
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$SLURM_ARRAY_TASK_ID" in
  0)
    SOURCE_ROOT="$REPO_DIR/data/processed/sdxl_trajectories_stacked/val"
    WORK_ROOT="$REPO_DIR/data/processed/sdxl_trajectories_cycle_r16_recap_coco_heldout"
    OUTPUT_DIR="$REPO_DIR/reports/eval/sdxl_recap_coco_heldout_cycle_r16"
    MAX_SAMPLES=1522
    ;;
  1)
    SOURCE_ROOT="$REPO_DIR/data/processed/sdxl_trajectories_pie_bench_ood/all"
    WORK_ROOT="$REPO_DIR/data/processed/sdxl_trajectories_pie_bench_ood/lora_r16_conditional_cycle"
    OUTPUT_DIR="$REPO_DIR/reports/eval/sdxl_pie_bench_ood/lora_r16_conditional_cycle"
    MAX_SAMPLES=700
    ;;
  *) echo "Unexpected evaluation task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-cycle-r16-eval-${SLURM_ARRAY_TASK_ID}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sdxl_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"

[[ -d $SOURCE_ROOT ]] || { echo "Source root not found: $SOURCE_ROOT" >&2; exit 2; }
if [[ ! -d $WORK_ROOT ]]; then
  mkdir -p "$(dirname "$WORK_ROOT")"
  cp -al "$SOURCE_ROOT" "$WORK_ROOT"
fi
MODEL_DIR="/dev/shm/${USER:-unknown}/sdxl-shared-model/stable-diffusion-xl-base-1.0"
stage_sdxl_model "$MODEL_DIR"

"$PYTHON" -m diff_inversion.eval.invert_sdxl input_dir="$WORK_ROOT" overwrite=true \
  lora.enabled=true lora.r=16 lora.lora_alpha=8 lora.checkpoint_path="$CHECKPOINT_PATH" \
  save_inversion_artifacts=false model_id_override="$MODEL_DIR"
"$PYTHON" -m diff_inversion.eval.reconstruct_sdxl input_dir="$WORK_ROOT" overwrite=true model_id_override="$MODEL_DIR"
"$PYTHON" -m diff_inversion.eval.run input_dir="$WORK_ROOT" output_dir="$OUTPUT_DIR" \
  max_samples="$MAX_SAMPLES" wandb.mode=disabled
