#!/bin/bash
#SBATCH --job-name=lora-all-editors
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=0-3%3
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

# One array task evaluates one editor, then measures it before exiting.
CHECKPOINT_REL=${1:?Pass checkpoint path relative to repository root}
RANK=${2:?Pass LoRA rank: 8, 16, or 32}
RUN_TAG=${3:?Pass a unique evaluation tag}
INVERSION_GUIDANCE=${4:-1.0}

case "$RANK" in 8|16|32) ;; *) echo "Unsupported rank: $RANK" >&2; exit 2 ;; esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=$REPO_DIR/$CHECKPOINT_REL
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve storage root" >&2; exit 2; }

MODULES=(diff_inversion.eval.p2p_lora diff_inversion.eval.pnp_lora diff_inversion.eval.masactrl_lora diff_inversion.eval.pix2pix_zero_lora)
CONFIGS=(eval/p2p_lora eval/pnp_lora eval/masactrl_lora eval/pix2pix_zero_lora)
METHODS=(lora+directinversion+p2p lora+directinversion+pnp lora+directinversion+masactrl lora+directinversion+pix2pix-zero)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODULE=${MODULES[$TASK_ID]:-}
CONFIG=${CONFIGS[$TASK_ID]:-}
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid array task: $TASK_ID" >&2; exit 2; }

EVAL_ROOT=$STORAGE_ROOT/evaluation/one_epoch/$RUN_TAG
CACHE_ROOT=$STORAGE_ROOT/cache/one_epoch/$RUN_TAG/${METHOD//+/_}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/lora-all-editors-${SLURM_ARRAY_JOB_ID}-${TASK_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export UV_CACHE_DIR=$STORAGE_ROOT/cache/uv
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$UV_CACHE_DIR" "$EVAL_ROOT" "$CACHE_ROOT"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m "$MODULE" --config-name "$CONFIG" \
  "edit_method_list=[$METHOD]" \
  lora="r$RANK" \
  lora.checkpoint_path="$CHECKPOINT" \
  output_path="$EVAL_ROOT" \
  cache_root="$CACHE_ROOT/editing" \
  guidance.inversion_scale="$INVERSION_GUIDANCE"

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  "method_name=$METHOD" \
  generated_root="$EVAL_ROOT" \
  metrics_root="$EVAL_ROOT/metrics" \
  log_root="$EVAL_ROOT/logs" \
  cache_root="$CACHE_ROOT/metrics"
