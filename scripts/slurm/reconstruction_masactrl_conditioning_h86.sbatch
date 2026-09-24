#!/bin/bash
#SBATCH --job-name=recon-masa-cond
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
SET=${1:-full}
case "$SET" in smoke|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac
REPO_DIR=${SLURM_SUBMIT_DIR:?}
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
DIAG_ROOT=$STORAGE_ROOT/evaluation/reconstruction_diagnostics
DATA_PATH=$([[ "$SET" == smoke ]] && echo "$DIAG_ROOT/manifests/smoke" || echo "$REPO_DIR/data/raw/PIE-Bench_v1")
OUTPUT_PATH=$DIAG_ROOT/$SET/masactrl_source_prompt
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/recon-masa-cond-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$OUTPUT_PATH"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev
cd pnp_inversion
python3 run_editing_masactrl.py --data_path "$DATA_PATH" --output_path "$OUTPUT_PATH" \
  --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
  --edit_method_list directinversion+masactrl \
  --model_key runwayml/stable-diffusion-v1-5 \
  --inversion_guidance_scale 1.0 --masactrl_inversion_prompt_mode source
