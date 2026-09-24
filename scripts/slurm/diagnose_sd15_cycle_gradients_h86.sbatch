#!/bin/bash
#SBATCH --job-name=sd15-cycle-grad
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-cycle-grad-${SLURM_JOB_ID}
VENV_DIR=${JOB_TMP_DIR}/venv
RESULT_PATH=logs/diagnostics/sd15_cycle_r16_lambda1_gradient_geometry_${SLURM_JOB_ID}.json

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${REPO_DIR}/scripts/slurm/job_local_cache.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "${JOB_TMP_DIR}"
}
trap cleanup EXIT
mkdir -p "${JOB_TMP_DIR}" "${REPO_DIR}/logs/diagnostics"

cd "${REPO_DIR}"
uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen

pytest -q tests/test_conditional_cycle_loss.py tests/test_cycle_gradient_diagnostics.py

python3 -m diff_inversion.modeling.cycle_gradient_diagnostics --config-name train_sd15 \
  lora=r16 \
  wandb=disabled \
  training_target=conditional_cycle \
  training_target.recon_lambda=1.0 \
  data.batch_size=4 \
  data.num_workers=0 \
  data.pin_memory=false \
  +diagnostics.num_trajectories=4 \
  +diagnostics.trained_checkpoint=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt \
  +diagnostics.output_path="${RESULT_PATH}"

echo "Cycle gradient diagnostic completed: ${RESULT_PATH}"
