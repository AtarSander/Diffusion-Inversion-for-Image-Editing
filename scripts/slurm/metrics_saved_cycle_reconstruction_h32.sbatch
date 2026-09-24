#!/bin/bash
# Metrics only. Existing PIE panels are:
# [instruction | source | standalone reconstruction | edited output].
# Usage: sbatch scripts/slurm/metrics_saved_cycle_reconstruction_h32.sbatch p2p
#SBATCH --job-name=cycle-recon-metrics
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail
EDITOR=${1:?Pass p2p, pnp, masactrl, or pix2pix-zero}
case "$EDITOR" in p2p|pnp|masactrl|pix2pix-zero) ;; *) exit 2 ;; esac
REPO_DIR=${SLURM_SUBMIT_DIR:?}
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
EVAL_ROOT=$STORAGE_ROOT/evaluation
METHOD=lora+directinversion+$EDITOR
OUTPUT_DIR=$EVAL_ROOT/reconstruction_diagnostics/full/saved_cycle_reconstruction_metrics/$EDITOR
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/cycle-recon-metrics-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
BASE_DIR=$EVAL_ROOT/sd15_fp16_baselines/directinversion+$EDITOR/annotation_images
R8_DIR=$EVAL_ROOT/one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors/$METHOD/annotation_images
R16_DIR=$EVAL_ROOT/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors/$METHOD/annotation_images
R32_DIR=$EVAL_ROOT/one_epoch/cycle_loss_cfg1_r32_uniform_e1_all_editors/$METHOD/annotation_images
for directory in "$BASE_DIR" "$R8_DIR" "$R16_DIR" "$R32_DIR"; do
  [[ -d "$directory" ]] || { echo "Missing saved panels: $directory" >&2; exit 2; }
done
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$OUTPUT_DIR"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev
python3 -m diff_inversion.eval.corrected_reconstruction_metrics \
  --mapping-file data/raw/PIE-Bench_v1/mapping_file.json \
  --source-root data/raw/PIE-Bench_v1/annotation_images \
  --method "directinversion=$BASE_DIR" \
  --method "cycle_lora_directinv_r8=$R8_DIR" \
  --method "cycle_lora_directinv_r16=$R16_DIR" \
  --method "cycle_lora_directinv_r32=$R32_DIR" \
  --output-csv "$OUTPUT_DIR/per_image.csv" \
  --summary-json "$OUTPUT_DIR/summary.json" \
  --device cuda --panel-index 2 --full-image
