#!/bin/bash
#SBATCH --job-name=lora-metrics
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}
RANK=${1:?Pass rank 8, 16, or 32}
case "$RANK" in
  8|16|32) ;;
  *) echo "Unsupported rank: $RANK (expected 8, 16, or 32)" >&2; exit 2 ;;
esac

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
JOB_TMP_ROOT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
JOB_TMP_DIR="$JOB_TMP_ROOT/lora-metrics-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
VENV_DIR="$JOB_TMP_DIR/venv"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REPO_DIR/.uv-cache}
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT

mkdir -p "$JOB_TMP_DIR" "$UV_CACHE_DIR"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

METHODS=(
  lora+p2p
  lora+directinversion+p2p
  lora+pnp
  lora+directinversion+pnp
  lora+masactrl
  lora+directinversion+masactrl
  lora+pix2pix-zero
  lora+directinversion+pix2pix-zero
)
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid task ID: $TASK_ID" >&2; exit 2; }

BASE=$REPO_DIR
python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  "method_name=$METHOD" \
  "generated_root=$BASE/evaluation/r$RANK" \
  "metrics_root=$BASE/evaluation/metrics/r$RANK" \
  "log_root=$BASE/logs/metrics/r$RANK" \
  "cache_root=$BASE/cache/h86/r$RANK/metrics"
