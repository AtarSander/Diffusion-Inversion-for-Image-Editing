#!/usr/bin/env bash
# Train the SDXL r16 conditional-cycle LoRA for the 15k-step baseline budget.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-48:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-64G}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-15000}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-2500}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_conditional_cycle_r16}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/checkpoints}

submit_segment() {
  local dependency=$1 segment=$2
  local args=(
    sbatch --parsable --job-name=sdxl-cycle-r16
    "--account=$SLURM_ACCOUNT" "--partition=$SLURM_PARTITION"
    --nodes=1 --ntasks=1 "--cpus-per-task=$SLURM_CPUS_PER_TASK" "--mem=$SLURM_MEM" "--gres=$SLURM_GRES"
    "--time=$SLURM_TIME" --signal=B:USR1@1800
    "--output=$RUN_LOG_DIR/segment_${segment}_%A.out"
    "--error=$RUN_LOG_DIR/segment_${segment}_%A.err"
    "--export=ALL,REPO_DIR=$REPO_DIR,PYTHON=$PYTHON,CHECKPOINT_ROOT=$CHECKPOINT_ROOT,MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS,SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS,SEGMENT=$segment"
  )
  [[ -n $dependency ]] && args+=("--dependency=afterok:$dependency")
  args+=("$0" --worker)
  "${args[@]}"
}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  FIRST_JOB=$(submit_segment "" 0)
  SECOND_JOB=$(submit_segment "$FIRST_JOB" 1)
  echo "Primary r16 15k job: $FIRST_JOB"
  echo "Fallback continuation: $SECOND_JOB"
  exit 0
fi

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-cycle-r16"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sdxl_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"

MODEL_DIR="/dev/shm/${USER:-unknown}/sdxl-shared-model/stable-diffusion-xl-base-1.0"
stage_sdxl_model "$MODEL_DIR"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/sdxl_inversion_lora_r16_conditional_cycle_lr5e-5_cosine_bf16"
mkdir -p "$CHECKPOINT_DIR"
STATE_PATH="$CHECKPOINT_DIR/training_state_final.pt"
RESUME_ARGS=()
if [[ -f $STATE_PATH ]]; then
  CURRENT_STEP=$("$PYTHON" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu")["global_step"]))' "$STATE_PATH")
  if (( CURRENT_STEP >= MAX_TRAIN_STEPS )); then
    echo "r16 already complete at step ${CURRENT_STEP}; no work for continuation ${SEGMENT}."
    exit 0
  fi
  RESUME_ARGS=(resume.enabled=true resume.mode=training_state resume.checkpoint_path="$STATE_PATH")
  echo "Resuming continuation ${SEGMENT} from step ${CURRENT_STEP}."
fi

"$PYTHON" -m diff_inversion.modeling.train --config-name train_sdxl_conditional_cycle \
  model.model_id="$MODEL_DIR" lora=r16 \
  run_name="sdxl-inversion-lora-r16-conditional-cycle-lr5e-5-cosine-bf16" \
  checkpoint_dir="$CHECKPOINT_DIR" max_train_steps="$MAX_TRAIN_STEPS" \
  save_every_steps="$SAVE_EVERY_STEPS" data.batch_size="$TRAIN_BATCH_SIZE" \
  gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" "${RESUME_ARGS[@]}"
