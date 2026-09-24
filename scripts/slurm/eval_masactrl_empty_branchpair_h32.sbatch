#!/bin/bash
#SBATCH --job-name=masa-empty
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
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

MODE=${1:?Pass base or lora_empty}
SET=${2:-smoke}
case "$MODE" in base|lora_empty) ;; *) echo "Invalid mode: $MODE" >&2; exit 2 ;; esac
case "$SET" in smoke|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT_REL=checkpoints/sd15_inversion_lora_branchpair_cfg1_r16_uniform_e1/checkpoint_final.pt
[[ -f "$REPO_DIR/$CHECKPOINT_REL" ]] || { echo "Missing checkpoint: $CHECKPOINT_REL" >&2; exit 2; }

if [[ "$SET" == smoke ]]; then
  DATA_PATH=checkpoints/../evaluation/reconstruction_diagnostics/manifests/smoke
else
  DATA_PATH=data/raw/PIE-Bench_v1
fi
OUTPUT_PATH=checkpoints/../evaluation/editor_compatibility/$SET/masactrl_empty_branchpair_cfg1_r16_uniform_e1

if [[ "$MODE" == base ]]; then
  METHOD=directinversion+masactrl
else
  METHOD=lora+directinversion+masactrl
fi

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/masa-empty-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "$JOB_TMP_DIR"
}
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

if [[ "$MODE" == base ]]; then
  python3 -m diff_inversion.eval.sd15_baseline \
    --config-name eval/sd15_baseline \
    backend=masactrl inversion=directinversion \
    data_path="$DATA_PATH" output_path="$OUTPUT_PATH" \
    cache_root="$JOB_CACHE_ROOT/editing"
else
  python3 -m diff_inversion.eval.masactrl_lora \
    --config-name eval/masactrl_lora \
    "edit_method_list=[$METHOD]" \
    lora=r16 lora.checkpoint_path="$CHECKPOINT_REL" lora.mode=branch_pair \
    data_path="$DATA_PATH" output_path="$OUTPUT_PATH" \
    cache_root="$JOB_CACHE_ROOT/editing" guidance.inversion_scale=1.0 \
    masactrl.lora_inversion_prompt_mode=empty
fi

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  method_name="$METHOD" data_path="$DATA_PATH" \
  generated_root="$OUTPUT_PATH" metrics_root="$OUTPUT_PATH/metrics" \
  log_root="$JOB_CACHE_ROOT/metric-logs" cache_root="$JOB_CACHE_ROOT/metrics"

echo "Completed matched MasaCtrl mode=$MODE set=$SET"
