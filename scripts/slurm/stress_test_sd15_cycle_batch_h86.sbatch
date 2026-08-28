#!/bin/bash
#SBATCH --job-name=sd15-cycle-cap
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "${REPO_DIR}/checkpoints")")"
RESULT_DIR=${STORAGE_ROOT}/logs/batch_size_stress
RESULT_LOG=${RESULT_DIR}/sd15_conditional_cycle_fp32_r32_${SLURM_JOB_ID}.log
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-cycle-cap-${SLURM_JOB_ID}
VENV_DIR=${JOB_TMP_DIR}/venv
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-128}
REQUIRED_BATCH_SIZE=${REQUIRED_BATCH_SIZE:-128}

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "${REPO_DIR}/scripts/slurm/job_local_cache.sh"

cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "${JOB_TMP_DIR}"
}
trap cleanup EXIT
mkdir -p "${RESULT_DIR}" "${JOB_TMP_DIR}"

cd "${REPO_DIR}"
uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen
pytest -q tests/test_conditional_cycle_loss.py

python3 -m diff_inversion.modeling.batch_size_stress_test --config-name train_sd15 \
  lora=r32 \
  training_target=conditional_cycle \
  training_target.recon_lambda=1.0 \
  model.torch_dtype=float32 \
  model.variant=null \
  +stress_test.synthetic=true \
  +stress_test.max_batch_size="${MAX_BATCH_SIZE}" >"${RESULT_LOG}" 2>&1

grep "largest_micro_batch" "${RESULT_LOG}"
CAPACITY=$(sed -n 's/.*largest_micro_batch=\([0-9][0-9]*\).*/\1/p' "${RESULT_LOG}" | tail -1)
[[ -n "${CAPACITY}" ]] || { echo "Could not parse batch capacity from ${RESULT_LOG}" >&2; exit 2; }
if (( CAPACITY < REQUIRED_BATCH_SIZE )); then
  echo "Cycle batch capacity ${CAPACITY} is below required batch ${REQUIRED_BATCH_SIZE}." >&2
  exit 3
fi

echo "Conditional-cycle batch ${REQUIRED_BATCH_SIZE} fits; full jobs may start."
echo "Full result log: ${RESULT_LOG}"
