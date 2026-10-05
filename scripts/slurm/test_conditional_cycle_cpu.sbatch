#!/bin/bash
#SBATCH --job-name=cycle-unit-tests
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/cycle-unit-tests-${SLURM_JOB_ID}
VENV_DIR=${JOB_TMP_DIR}/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
source "${REPO_DIR}/scripts/slurm/job_local_cache.sh"

cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "${JOB_TMP_DIR}"
}
trap cleanup EXIT
mkdir -p "${JOB_TMP_DIR}"

cd "${REPO_DIR}"
uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen

pytest -q tests/test_conditional_cycle_loss.py
