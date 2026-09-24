#!/bin/bash
#SBATCH --job-name=pix-grad-diag
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

SET=${1:-smoke}
case "$SET" in smoke|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt
[[ -f "$REPO_DIR/$CHECKPOINT_REL" ]] || { echo "Missing checkpoint: $CHECKPOINT_REL" >&2; exit 2; }

if [[ "$SET" == smoke ]]; then
  DATA_PATH=checkpoints/../evaluation/reconstruction_diagnostics/manifests/smoke
  MAX_IMAGES=1
else
  DATA_PATH=checkpoints/../evaluation/reconstruction_diagnostics/manifests/pix2pix_worst20
  MAX_IMAGES=20
fi
OUTPUT_PATH=checkpoints/../evaluation/editor_compatibility/diagnostics/pix2pix_attention_gradients_scaled/$SET

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/pix-grad-diag-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "$JOB_TMP_DIR"
}
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$OUTPUT_PATH"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

cd pnp_inversion
python3 run_pix2pix_attention_gradient_diagnostic.py \
  --data_path "$REPO_DIR/$DATA_PATH" \
  --output_path "$REPO_DIR/$OUTPUT_PATH" \
  --lora_checkpoint "$REPO_DIR/$CHECKPOINT_REL" \
  --lora_rank 16 --max_images "$MAX_IMAGES"

echo "Completed Pix2Pix-Zero attention-gradient diagnostic set=$SET"
