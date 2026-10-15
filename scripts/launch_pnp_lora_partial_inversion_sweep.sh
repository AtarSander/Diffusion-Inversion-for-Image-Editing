#!/usr/bin/env bash
# Nine PIE-Bench PnP runs, each followed by its complete metric suite.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
METRICS_PYTHON=${METRICS_PYTHON:-$REPO_DIR/.venv-pnp-metrics/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
LOG_ROOT=${LOG_ROOT:-$REPO_DIR/data/logs/slurm/pnp_lora_partial_inversion}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/data/evaluation/pnp_lora_partial_inversion_r16}
CHECKPOINT=${CHECKPOINT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/checkpoints/sd15_inversion_lora_r16_lr5e-5_cosine_bf16/checkpoint_step_130000.pt}
PIE_BENCH_DATA=${PIE_BENCH_DATA:-$REPO_DIR/data/raw/PIE-Bench_v1}
MODEL_CACHE=${MODEL_CACHE:-$REPO_DIR/data/model_cache/piebench/pnp}

if [[ ${1:-} != --worker ]]; then
  [[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
  [[ -f "$PIE_BENCH_DATA/mapping_file.json" ]] || { echo "Missing PIE-Bench mapping" >&2; exit 2; }
  [[ -x "$PYTHON" && -x "$METRICS_PYTHON" ]] || { echo "Missing Python environment" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=pnp-lora-partial --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=12 \
    --mem=64G --gres=gpu:1 --time=24:00:00 --array=10-90:10 \
    --output="$RUN_LOG_DIR/%A_%a.out" --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",METRICS_PYTHON="$METRICS_PYTHON",OUTPUT_ROOT="$OUTPUT_ROOT",CHECKPOINT="$CHECKPOINT",PIE_BENCH_DATA="$PIE_BENCH_DATA",MODEL_CACHE="$MODEL_CACHE" \
    "$0" --worker
fi

PERCENT=${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}
(( PERCENT >= 10 && PERCENT <= 90 && PERCENT % 10 == 0 )) || {
  echo "Invalid sweep percentage: $PERCENT" >&2
  exit 2
}
INVERSION_STEPS=$((PERCENT / 2))
RUN_ROOT="$OUTPUT_ROOT/p${PERCENT}"
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_ARRAY_JOB_ID}-${PERCENT}}"
JOB_TMP_DIR="$SLURM_TMPDIR/pnp-lora-partial"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
cd "$REPO_DIR"

"$PYTHON" -m diff_inversion.eval.pnp_lora --config-name eval/pnp_lora \
  lora=r16 lora.checkpoint_path="$CHECKPOINT" \
  data_path="$PIE_BENCH_DATA" output_path="$RUN_ROOT" \
  cache_root="$MODEL_CACHE" edit_method_list='[lora+pnp]' \
  inversion_steps="$INVERSION_STEPS" hydra.run.dir="$RUN_ROOT/logs/editing"

"$PYTHON" -m diff_inversion.eval.pnp_lora_metrics --config-name eval/pnp_lora_metrics \
  data_path="$PIE_BENCH_DATA" generated_root="$RUN_ROOT" \
  metrics_root="$RUN_ROOT/metrics" log_root="$RUN_ROOT/logs/metrics" \
  cache_root="$MODEL_CACHE/metrics" +python="$METRICS_PYTHON" method_name=lora+pnp
