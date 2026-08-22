#!/bin/bash
#SBATCH --job-name=pnp-lora-chain
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

CHECKPOINT=${1:?Pass an adapter checkpoint path}
RANK=${2:?Pass LoRA rank 8, 16, or 32}
RUN_TAG=${3:?Pass a unique evaluation tag}
INVERSION_GUIDANCE=${4:-1.0}
METRICS_ONLY=${5:-false}
case "$RANK" in
  8|16|32) ;;
  *) echo "Unsupported rank: $RANK (expected 8, 16, or 32)" >&2; exit 2 ;;
esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve project storage from $REPO_DIR/checkpoints" >&2; exit 2; }
SCRATCH_ROOT="$STORAGE_ROOT"
EVAL_ROOT="$SCRATCH_ROOT/evaluation/pnp_checkpoints/$RUN_TAG"
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/pnp-lora-chain-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint does not exist: $CHECKPOINT" >&2; exit 2; }

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

if [[ "$METRICS_ONLY" != true && "$METRICS_ONLY" != 1 ]]; then
python3 -m diff_inversion.eval.pnp_lora --config-name eval/pnp_lora lora="r$RANK" lora.checkpoint_path="$CHECKPOINT" output_path="$EVAL_ROOT" cache_root="$JOB_CACHE_ROOT/editing" guidance.inversion_scale="$INVERSION_GUIDANCE"
fi

for method in "lora+pnp" "lora+directinversion+pnp"; do
  python3 -m diff_inversion.eval.pnp_lora_metrics --config-name eval/pnp_lora_metrics "method_name=$method" generated_root="$EVAL_ROOT" metrics_root="$EVAL_ROOT/metrics" log_root="$EVAL_ROOT/logs" cache_root="$JOB_CACHE_ROOT/metrics"
done
