#!/bin/bash
#SBATCH --job-name=cycle-eval-smoke
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

# One PIE-Bench image, both pure-LoRA and LoRA+DirectInv P2P paths, then metrics.
CHECKPOINT_REL=${1:?Pass checkpoint path relative to repository root}
RANK=${2:?Pass LoRA rank: 8, 16, or 32}
RUN_TAG=${3:-cycle_loss_smoke}

case "$RANK" in 8|16|32) ;; *) echo "Unsupported rank: $RANK" >&2; exit 2 ;; esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=$REPO_DIR/$CHECKPOINT_REL
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
SOURCE_DATA=$REPO_DIR/data/raw/PIE-Bench_v1
EVAL_ROOT=$STORAGE_ROOT/evaluation/smoke/$RUN_TAG
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/cycle-eval-smoke-${SLURM_JOB_ID}
SMOKE_DATA=$JOB_TMP_DIR/pie_bench_one
VENV_DIR=$JOB_TMP_DIR/venv

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$SOURCE_DATA/mapping_file.json" ]] || { echo "Missing PIE-Bench mapping" >&2; exit 2; }

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$SMOKE_DATA" "$EVAL_ROOT"
ln -s "$SOURCE_DATA/annotation_images" "$SMOKE_DATA/annotation_images"
jq 'with_entries(select(.key == "000000000000"))' \
  "$SOURCE_DATA/mapping_file.json" > "$SMOKE_DATA/mapping_file.json"
[[ $(jq 'length' "$SMOKE_DATA/mapping_file.json") -eq 1 ]] || {
  echo "Failed to construct one-image smoke mapping" >&2
  exit 2
}

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.eval.p2p_lora --config-name eval/p2p_lora \
  'edit_method_list=[lora+p2p,lora+directinversion+p2p]' \
  'edit_category_list=["0"]' \
  lora="r$RANK" \
  lora.checkpoint_path="$CHECKPOINT" \
  lora.mode=single \
  data_path="$SMOKE_DATA" \
  output_path="$EVAL_ROOT" \
  cache_root="$JOB_CACHE_ROOT/editing" \
  guidance.inversion_scale=1.0

for method in lora+p2p lora+directinversion+p2p; do
  python3 -m diff_inversion.eval.pnp_lora_metrics \
    --config-name eval/pnp_lora_metrics \
    "method_name=$method" \
    'edit_category_list=["0"]' \
    data_path="$SMOKE_DATA" \
    generated_root="$EVAL_ROOT" \
    metrics_root="$EVAL_ROOT/metrics" \
    log_root="$EVAL_ROOT/logs" \
    cache_root="$JOB_CACHE_ROOT/metrics"
done

echo "Cycle-loss LoRA evaluation smoke passed for rank $RANK."
