#!/usr/bin/env bash
# Train SDXL LoRA r8/r16/r32 with checkpoint-safe conditional-cycle continuations.
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
CONTINUATION_WINDOWS=${CONTINUATION_WINDOWS:-30}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-171225}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-2500}
SMOKE_STEPS=${SMOKE_STEPS:-100}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_conditional_cycle}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/checkpoints}

submit_job() {
  local dependency=$1 segment=$2 worker_mode=$3 output_prefix=$4
  local args=(
    sbatch --parsable --job-name=sdxl-cycle-train
    "--account=$SLURM_ACCOUNT" "--partition=$SLURM_PARTITION"
    --nodes=1 --ntasks=1 "--cpus-per-task=$SLURM_CPUS_PER_TASK" "--mem=$SLURM_MEM" "--gres=$SLURM_GRES"
    "--time=$SLURM_TIME" --signal=B:USR1@1800
    "--output=$RUN_LOG_DIR/${output_prefix}_%A_%a.out"
    "--error=$RUN_LOG_DIR/${output_prefix}_%A_%a.err"
    "--export=ALL,REPO_DIR=$REPO_DIR,PYTHON=$PYTHON,CHECKPOINT_ROOT=$CHECKPOINT_ROOT,MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS,SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS,SEGMENT=$segment"
  )
  [[ -n $dependency ]] && args+=("--dependency=afterok:$dependency")
  if [[ $worker_mode == --smoke-worker ]]; then
    args+=(--array=0)
  else
    args+=(--array=0-2)
  fi
  args+=("$0" "$worker_mode")
  "${args[@]}"
}

if [[ ${1:-} != --worker && ${1:-} != --smoke-worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  SMOKE_JOB=$(submit_job "" smoke --smoke-worker smoke)
  PREVIOUS_JOB=$SMOKE_JOB
  for ((segment = 0; segment < CONTINUATION_WINDOWS; segment++)); do
    PREVIOUS_JOB=$(submit_job "$PREVIOUS_JOB" "$segment" --worker "segment_${segment}")
    echo "Continuation $segment: $PREVIOUS_JOB"
  done
  echo "Cycle-loss smoke: $SMOKE_JOB"
  echo "Final continuation array: $PREVIOUS_JOB"
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ ${1:-} == --smoke-worker ]]; then
  RANK=32
else
  case "$SLURM_ARRAY_TASK_ID" in
    0) RANK=8 ;;
    1) RANK=16 ;;
    2) RANK=32 ;;
    *) echo "Unexpected training task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
  esac
fi

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-cycle-train-r${RANK}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
if [[ ${1:-} == --smoke-worker ]]; then
  CHECKPOINT_DIR="$JOB_TMP_DIR/smoke-checkpoints"
else
  CHECKPOINT_DIR="$CHECKPOINT_ROOT/sdxl_inversion_lora_r${RANK}_conditional_cycle_lr5e-5_cosine_bf16"
fi
mkdir -p "$CHECKPOINT_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sdxl_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"

MODEL_DIR="/dev/shm/${USER:-unknown}/sdxl-shared-model/stable-diffusion-xl-base-1.0"
stage_sdxl_model "$MODEL_DIR"
if [[ ${1:-} == --smoke-worker ]]; then
  "$PYTHON" -m diff_inversion.modeling.train --config-name train_sdxl_conditional_cycle \
    model.model_id="$MODEL_DIR" lora=r32 wandb.mode=disabled max_train_steps="$SMOKE_STEPS" \
    eval_every_steps=999999 save_every_steps=999999 checkpoint_dir="$CHECKPOINT_DIR"
  exit 0
fi

STATE_PATH="$CHECKPOINT_DIR/training_state_final.pt"
RESUME_ARGS=()
if [[ -f $STATE_PATH ]]; then
  CURRENT_STEP=$("$PYTHON" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu")["global_step"]))' "$STATE_PATH")
  if (( CURRENT_STEP >= MAX_TRAIN_STEPS )); then
    echo "r${RANK} already complete at step ${CURRENT_STEP}; no work for continuation ${SEGMENT}."
    exit 0
  fi
  RESUME_ARGS=(resume.enabled=true resume.mode=training_state resume.checkpoint_path="$STATE_PATH")
  echo "Resuming r${RANK} continuation ${SEGMENT} from step ${CURRENT_STEP}."
fi

"$PYTHON" -m diff_inversion.modeling.train --config-name train_sdxl_conditional_cycle \
  model.model_id="$MODEL_DIR" lora="r${RANK}" \
  run_name="sdxl-inversion-lora-r${RANK}-conditional-cycle-lr5e-5-cosine-bf16" \
  checkpoint_dir="$CHECKPOINT_DIR" max_train_steps="$MAX_TRAIN_STEPS" \
  save_every_steps="$SAVE_EVERY_STEPS" "${RESUME_ARGS[@]}"
