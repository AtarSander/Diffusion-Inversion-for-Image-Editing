#!/bin/bash
#SBATCH --job-name=lora-split-metrics
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

RUN_TAG=${1:?Pass a complete evaluation tag}
SOURCE_GROUP=${2:?Pass real, generated, or unknown}
case "$SOURCE_GROUP" in
  real|generated|unknown) ;;
  *) echo "Unsupported source group: $SOURCE_GROUP" >&2; exit 2 ;;
esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
EVAL_ROOT="$STORAGE_ROOT/evaluation/one_epoch/$RUN_TAG"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

METHODS=(
  "lora+directinversion+p2p"
  "lora+directinversion+pnp"
  "lora+directinversion+masactrl"
  "lora+directinversion+pix2pix-zero"
)
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid array task: $TASK_ID" >&2; exit 2; }
GENERATED_DIR="$EVAL_ROOT/$METHOD/annotation_images"
[[ -d "$GENERATED_DIR" ]] || { echo "Missing generated images: $GENERATED_DIR" >&2; exit 2; }

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/lora-split-metrics-${SLURM_ARRAY_JOB_ID}-${TASK_ID}
VENV_DIR="$JOB_TMP_DIR/venv"

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "$JOB_TMP_DIR"
}
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$EVAL_ROOT/metrics/$SOURCE_GROUP" "$EVAL_ROOT/logs/$SOURCE_GROUP"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  "method_name=$METHOD" \
  "source_group=$SOURCE_GROUP" \
  "generated_root=$EVAL_ROOT" \
  "metrics_root=$EVAL_ROOT/metrics/$SOURCE_GROUP" \
  "log_root=$EVAL_ROOT/logs/$SOURCE_GROUP" \
  "cache_root=$JOB_CACHE_ROOT/metrics"
