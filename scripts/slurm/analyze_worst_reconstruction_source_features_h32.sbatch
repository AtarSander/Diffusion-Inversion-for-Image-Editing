#!/bin/bash
#SBATCH --job-name=worst-src-feat
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com
set -euo pipefail
REPO_DIR=${SLURM_SUBMIT_DIR:?}; JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/worst-src-feat-${SLURM_JOB_ID}; VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15; module load uv/0.11.2
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup(){ [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }; trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"; cd "$REPO_DIR"; uv venv --python "$(command -v python3)" "$VENV_DIR"; source "$VENV_DIR/bin/activate"; uv sync --active --frozen --no-dev
python3 scripts/analysis/analyze_worst_reconstruction_source_features.py --data data/raw/PIE-Bench_v1 --metrics-root checkpoints/../evaluation/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors/metrics --selection-root checkpoints/../evaluation/editor_compatibility/selected_reconstruction_examples/cycle_r16_uniform_complete_metric_contribution --output checkpoints/../evaluation/editor_compatibility/selected_reconstruction_examples/cycle_r16_uniform_complete_metric_contribution/source_feature_analysis.json
