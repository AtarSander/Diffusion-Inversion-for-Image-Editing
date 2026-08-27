#!/bin/bash
#SBATCH --job-name=sd15-base-fp16
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

# Submit as eight ordinary jobs (task IDs 0-7), never as a Slurm array.
TASK_ID=${1:?Pass task ID 0-7}
BACKENDS=(p2p p2p pnp pnp masactrl masactrl pix2pix_zero pix2pix_zero)
INVERSIONS=(ddim directinversion ddim directinversion ddim directinversion ddim directinversion)
METHODS=(ddim+p2p directinversion+p2p ddim+pnp directinversion+pnp ddim+masactrl directinversion+masactrl ddim+pix2pix-zero directinversion+pix2pix-zero)

BACKEND=${BACKENDS[$TASK_ID]:-}
INVERSION=${INVERSIONS[$TASK_ID]:-}
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid task ID: $TASK_ID (expected 0-7)" >&2; exit 2; }

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-base-fp16-${SLURM_JOB_ID}-${TASK_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "$JOB_TMP_DIR"
}
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.eval.sd15_baseline \
  --config-name eval/sd15_baseline \
  backend="$BACKEND" \
  inversion="$INVERSION"

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/sd15_baseline_metrics \
  "method_name=$METHOD"
