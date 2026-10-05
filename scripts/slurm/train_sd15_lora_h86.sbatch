#!/bin/bash
#SBATCH --job-name=sd15-lora
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

RANK=${1:?Pass rank 8, 16, or 32}
case "$RANK" in
  8|16|32) ;;
  *) echo "Unsupported rank: $RANK (expected 8, 16, or 32)" >&2; exit 2 ;;
esac
shift
HYDRA_OVERRIDES=("$@")
WANDB_PROFILE=online
TRAIN_VARIANT=uniform
CHECKPOINT_DIR_OVERRIDE=
for override in "${HYDRA_OVERRIDES[@]}"; do
  case "$override" in
    wandb=offline|wandb.mode=offline) WANDB_PROFILE=offline ;;
    wandb=disabled|wandb.mode=disabled) WANDB_PROFILE=disabled ;;
    wandb=online|wandb.mode=online) WANDB_PROFILE=online ;;
    sampling=final_tail|sampling.mode=final_tail) TRAIN_VARIANT=final_tail ;;
    checkpoint_dir=*) CHECKPOINT_DIR_OVERRIDE=${override#checkpoint_dir=} ;;
  esac
done

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve project storage from $REPO_DIR/checkpoints" >&2; exit 2; }
RUN_NAME=sd15-inversion-lora-r${RANK}-${TRAIN_VARIANT}-e1
CHECKPOINT_DIR=$REPO_DIR/checkpoints/sd15_inversion_lora_r${RANK}_${TRAIN_VARIANT}_e1
if [[ -n "$CHECKPOINT_DIR_OVERRIDE" ]]; then
  if [[ "$CHECKPOINT_DIR_OVERRIDE" = /* ]]; then
    CHECKPOINT_DIR=$CHECKPOINT_DIR_OVERRIDE
  else
    CHECKPOINT_DIR=$REPO_DIR/$CHECKPOINT_DIR_OVERRIDE
  fi
fi
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-lora-r${RANK}-${TRAIN_VARIANT}-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$CHECKPOINT_DIR"

RESUME_OVERRIDES=()
if [[ "${AUTO_RESUME:-0}" == "1" ]]; then
  if [[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" ]]; then
    echo "Final checkpoint already exists; nothing to resume: $CHECKPOINT_DIR/checkpoint_final.pt"
    exit 0
  fi

  LATEST_STATE=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
    -name 'training_state_step_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  if [[ -z "$LATEST_STATE" ]]; then
    echo "AUTO_RESUME=1 but no periodic training-state checkpoint exists in $CHECKPOINT_DIR" >&2
    exit 3
  fi

  RESUME_PATH=$CHECKPOINT_DIR/$LATEST_STATE
  RESUME_OVERRIDES=(
    resume.enabled=true
    resume.mode=training_state
    "resume.checkpoint_path=$RESUME_PATH"
  )
  echo "Automatically resuming from $RESUME_PATH"
fi

cd "$REPO_DIR"

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi
if [[ "$WANDB_PROFILE" == online && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing; add it to $REPO_DIR/.env" >&2
  exit 2
fi

uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

echo "Training rank $RANK, variant $TRAIN_VARIANT."
python3 -m diff_inversion.modeling.train --config-name train_sd15 lora="r$RANK" run_name="$RUN_NAME" checkpoint_dir="$CHECKPOINT_DIR" \
  "${HYDRA_OVERRIDES[@]}" "${RESUME_OVERRIDES[@]}"
