#!/bin/bash
#SBATCH --job-name=sdxl-tada-metrics
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

EXPECTED_SAMPLES=${1:-1522}
OUTPUT_ROOT=${2:-reports/eval/sdxl_tada_matched}
REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sdxl-tada-metrics-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.eval.tada_matched_metrics --config-name eval/tada_matched_metrics \
  expected_samples="$EXPECTED_SAMPLES" output_root="$OUTPUT_ROOT"

if [[ "$EXPECTED_SAMPLES" == "1522" ]]; then
  python3 -m scripts.analysis.summarize_sdxl_tada_matched
  python3 -m scripts.analysis.plot_iclr_experiments
fi
