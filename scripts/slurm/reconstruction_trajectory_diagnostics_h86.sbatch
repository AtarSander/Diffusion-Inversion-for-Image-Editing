#!/bin/bash
#SBATCH --job-name=recon-trajectory
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
MAX_IMAGES=${1:-0}
TAG=${2:-full}
REPO_DIR=${SLURM_SUBMIT_DIR:?}
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
DIAG_ROOT=$STORAGE_ROOT/evaluation/reconstruction_diagnostics/$TAG/trajectory
CHECKPOINT=$REPO_DIR/checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/recon-trajectory-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$DIAG_ROOT"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev
cd pnp_inversion
python3 run_trajectory_diagnostics.py \
  --data_path "$REPO_DIR/data/raw/PIE-Bench_v1" \
  --output_jsonl "$DIAG_ROOT/per_image.jsonl" --summary_json "$DIAG_ROOT/summary.json" \
  --lora_checkpoint "$CHECKPOINT" --lora_rank 16 --max_images "$MAX_IMAGES" \
  --baseline_metrics_root "$STORAGE_ROOT/evaluation/sd15_fp16_baselines/metrics" \
  --experiment_metrics_root "$STORAGE_ROOT/evaluation/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors/metrics"
