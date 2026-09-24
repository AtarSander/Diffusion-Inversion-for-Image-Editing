#!/bin/bash
#SBATCH --job-name=sd15-endpoint-r16
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
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

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
RUN_NAME=sd15-one-step-endpoint-r16-e1
CHECKPOINT_DIR=${REPO_DIR}/checkpoints/sd15_one_step_endpoint_r16_e1
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-one-step-endpoint-r16-${SLURM_JOB_ID}
VENV_DIR=${JOB_TMP_DIR}/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${REPO_DIR}/scripts/slurm/job_local_cache.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "${JOB_TMP_DIR}"; }
trap cleanup EXIT
mkdir -p "${JOB_TMP_DIR}" "${CHECKPOINT_DIR}"
cd "${REPO_DIR}"

if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  source "${REPO_DIR}/.env"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing; add it to ${REPO_DIR}/.env" >&2
  exit 2
fi

uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.modeling.train --config-name train_sd15 \
  lora=r16 training_target=conditional_endpoint \
  run_name="${RUN_NAME}" checkpoint_dir="${CHECKPOINT_DIR}" \
  max_train_steps=11892 gradient_accumulation_steps=1
