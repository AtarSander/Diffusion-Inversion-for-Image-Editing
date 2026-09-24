#!/bin/bash
#SBATCH --job-name=sdxl-tada
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

START_INDEX=${1:?Pass the first prompt index}
NUM_SAMPLES=${2:?Pass the number of prompts}
REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sdxl-tada-${SLURM_JOB_ID}
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

python3 -m diff_inversion.data.generate_sdxl_samples \
  --config-name sample_gather_sdxl_tada \
  start_index="$START_INDEX" \
  num_samples="$NUM_SAMPLES"

python3 -m diff_inversion.eval.tada_sdxl \
  --config-name eval/tada_sdxl \
  start_index="$START_INDEX" \
  num_samples="$NUM_SAMPLES"
