#!/bin/bash
#SBATCH --job-name=sd15-cycle-smoke
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
BATCH_SIZE=${BATCH_SIZE:-1}
RECON_LAMBDA=${RECON_LAMBDA:-1.0}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-cycle-smoke-${SLURM_JOB_ID}
VENV_DIR=${JOB_TMP_DIR}/venv
CHECKPOINT_DIR=${JOB_TMP_DIR}/checkpoints

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
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
mkdir -p "${JOB_TMP_DIR}" "${CHECKPOINT_DIR}"

cd "${REPO_DIR}"
uv venv --python "$(command -v python3)" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
uv sync --active --frozen --no-dev

python3 -m diff_inversion.modeling.train --config-name train_sd15 \
  lora=r16 \
  wandb=disabled \
  training_target=conditional_cycle \
  training_target.recon_lambda="${RECON_LAMBDA}" \
  data.batch_size="${BATCH_SIZE}" \
  data.num_workers=0 \
  data.pin_memory=false \
  max_train_steps=1 \
  gradient_accumulation_steps=1 \
  log_every_steps=1 \
  eval_every_steps=0 \
  save_every_steps=0 \
  max_val_batches=0 \
  validation_preview.enabled=false \
  save_training_state=false \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  run_name=sd15-conditional-cycle-smoke

echo "Conditional-cycle one-step GPU smoke passed."
