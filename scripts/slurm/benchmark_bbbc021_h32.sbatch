#!/usr/bin/env bash
#SBATCH --job-name=bbbc021-lora
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com
set -euo pipefail
REPO_DIR=${SLURM_SUBMIT_DIR:?}
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/bbbc021-${SLURM_JOB_ID}
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4 WANDB_MODE=disabled
export PYTHONPYCACHEPREFIX="$JOB_TMP_DIR/pycache"
cd "$REPO_DIR"
sha256sum diff_inversion/eval/bbbc021.py diff_inversion/eval/ovg_bridge.py
bash scripts/setup_ovg.sh
uv venv --python "$(command -v python3)" "$JOB_TMP_DIR/venv"
source "$JOB_TMP_DIR/venv/bin/activate"
uv sync --active --frozen --no-dev
# OVG's package metadata pins incompatible Torch/xformers versions and does not
# package its root sources correctly. Import the pinned source via our bridge.
uv pip install --python "$VIRTUAL_ENV/bin/python" h5py==3.14.0 torch-fidelity==0.3.0 pytest==9.0.2
if [[ "${1:-}" == "--smoke" ]]; then
  python -m pytest -q tests/test_ovg_bridge.py
  python -m diff_inversion.eval.bbbc021 --cfg job
else
  python -m diff_inversion.eval.bbbc021 "$@"
fi
