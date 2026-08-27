#!/bin/bash
#SBATCH --job-name=lora-full-editors
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
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

# One ordinary job runs one editor/method and then calculates its PIE-Bench metrics.
CHECKPOINT_REL=${1:?Pass checkpoint path relative to repository root}
RANK=${2:?Pass LoRA rank: 8, 16, or 32}
RUN_TAG=${3:?Pass a unique evaluation tag}
TASK_ID=${4:?Pass method task id 0-7}
INVERSION_GUIDANCE=${5:-1.0}
LORA_MODE=${6:-single}

case "$RANK" in 8|16|32) ;; *) echo "Unsupported rank: $RANK" >&2; exit 2 ;; esac
case "$LORA_MODE" in single|branch_pair) ;; *) echo "Unsupported LoRA mode: $LORA_MODE" >&2; exit 2 ;; esac
if [[ "$LORA_MODE" == branch_pair && $((TASK_ID % 2)) -eq 0 ]]; then
  echo "Branch-pair checkpoints are supported only for Direct-Inversion tasks 1, 3, 5, and 7." >&2
  exit 2
fi

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=$REPO_DIR/$CHECKPOINT_REL
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve storage root" >&2; exit 2; }

MODULES=(
  diff_inversion.eval.p2p_lora diff_inversion.eval.p2p_lora
  diff_inversion.eval.pnp_lora diff_inversion.eval.pnp_lora
  diff_inversion.eval.masactrl_lora diff_inversion.eval.masactrl_lora
  diff_inversion.eval.pix2pix_zero_lora diff_inversion.eval.pix2pix_zero_lora
)
CONFIGS=(
  eval/p2p_lora eval/p2p_lora
  eval/pnp_lora eval/pnp_lora
  eval/masactrl_lora eval/masactrl_lora
  eval/pix2pix_zero_lora eval/pix2pix_zero_lora
)
METHODS=(
  lora+p2p lora+directinversion+p2p
  lora+pnp lora+directinversion+pnp
  lora+masactrl lora+directinversion+masactrl
  lora+pix2pix-zero lora+directinversion+pix2pix-zero
)

MODULE=${MODULES[$TASK_ID]:-}
CONFIG=${CONFIGS[$TASK_ID]:-}
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid method task: $TASK_ID" >&2; exit 2; }

EVAL_ROOT=$STORAGE_ROOT/evaluation/one_epoch/$RUN_TAG
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/lora-full-editors-${SLURM_JOB_ID}-${TASK_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$EVAL_ROOT"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m "$MODULE" --config-name "$CONFIG" \
  "edit_method_list=[$METHOD]" \
  lora="r$RANK" \
  lora.checkpoint_path="$CHECKPOINT" \
  lora.mode="$LORA_MODE" \
  output_path="$EVAL_ROOT" \
  cache_root="$JOB_CACHE_ROOT/editing" \
  guidance.inversion_scale="$INVERSION_GUIDANCE"

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  "method_name=$METHOD" \
  generated_root="$EVAL_ROOT" \
  metrics_root="$EVAL_ROOT/metrics" \
  log_root="$EVAL_ROOT/logs" \
  cache_root="$JOB_CACHE_ROOT/metrics"
