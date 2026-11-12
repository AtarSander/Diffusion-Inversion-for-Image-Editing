#!/usr/bin/env bash
#SBATCH --job-name=edges2shoes-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/edges2shoes-data-${SLURM_JOB_ID}
source scripts/slurm/job_local_cache.sh
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
module load python/3.11.15 uv/0.11.2
export PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="$JOB_TMP_DIR/pycache"
uv venv --python "$(command -v python3)" "$JOB_TMP_DIR/venv"
uv pip install --python "$JOB_TMP_DIR/venv/bin/python" numpy==2.2.6 Pillow==11.3.0 PyYAML==6.0.2
"$JOB_TMP_DIR/venv/bin/python" scripts/prepare_edges2shoes.py
