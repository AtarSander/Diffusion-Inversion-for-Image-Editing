#!/usr/bin/env bash
# Submit the full PIE-Bench editor suite after successful mixture training.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sd15_recap80_diffdb20_piebench}
OUTPUT_ROOT=${OUTPUT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/evaluation/sd15_recap80_diffdb20_r16_fp32_mb64_accum2}
CHECKPOINT=${CHECKPOINT:-$REPO_DIR/checkpoints/sd15_inversion_lora_r16_recap80_diffusiondb20_fp32_mb64_accum2/checkpoint_final.pt}
PIE_BENCH_DATA=${PIE_BENCH_DATA:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/raw/PIE-Bench_v1}
SLURM_TMP_MB=${SLURM_TMP_MB:-20000}
SLURM_ARRAY=${SLURM_ARRAY:-0-3}
PERSISTENT_MODEL_CACHE=${PERSISTENT_MODEL_CACHE:-/net/tscratch/people/plgatarsander/ZZSN_data/model_cache/piebench}
TRAIN_JOB_ID=${1:?Pass the successful mixture-training Slurm job ID.}
[[ -f "$PIE_BENCH_DATA/mapping_file.json" ]] || { echo "Missing PIE-Bench mapping file: $PIE_BENCH_DATA/mapping_file.json" >&2; exit 2; }

if [[ ${2:-} != --worker ]]; then
  if [[ ${SKIP_DEPENDENCY:-false} == true ]]; then
    DEPENDENCY_ARGS=()
  else
    DEPENDENCY_ARGS=(--dependency="afterok:${TRAIN_JOB_ID}")
  fi
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=piebench-sd15-mixture --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=32G --gres=gpu:1 --time=13:00:00 \
    "${DEPENDENCY_ARGS[@]}" --array="$SLURM_ARRAY" \
    --output="$RUN_LOG_DIR/%A_%a.out" --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",OUTPUT_ROOT="$OUTPUT_ROOT",CHECKPOINT="$CHECKPOINT",PIE_BENCH_DATA="$PIE_BENCH_DATA",PERSISTENT_MODEL_CACHE="$PERSISTENT_MODEL_CACHE" \
    "$0" "$TRAIN_JOB_ID" --worker
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODULES=(diff_inversion.eval.p2p_lora diff_inversion.eval.pnp_lora diff_inversion.eval.masactrl_lora diff_inversion.eval.pix2pix_zero_lora)
CONFIGS=(eval/p2p_lora eval/pnp_lora eval/masactrl_lora eval/pix2pix_zero_lora)
NAMES=(p2p pnp masactrl pix2pix_zero)
MODULE=${MODULES[$TASK_ID]:-}
CONFIG=${CONFIGS[$TASK_ID]:-}
NAME=${NAMES[$TASK_ID]:-}
[[ -n "$MODULE" ]] || { echo "Invalid array task: $TASK_ID" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing final mixture checkpoint: $CHECKPOINT" >&2; exit 2; }

export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_ARRAY_JOB_ID}-${TASK_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/piebench-sd15-mixture"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
cd "$REPO_DIR"
[[ -f .env ]] && { set -a; source .env; set +a; }
"$PYTHON" -m "$MODULE" --config-name "$CONFIG" \
  lora=r16 lora.checkpoint_path="$CHECKPOINT" \
  data_path="$PIE_BENCH_DATA" output_path="$OUTPUT_ROOT/$NAME" cache_root="$PERSISTENT_MODEL_CACHE/$NAME"

# Score both inversion variants once this editor has generated all of its images.
case "$NAME" in
  p2p) METRIC_METHODS=(lora+p2p lora+directinversion+p2p) ;;
  pnp) METRIC_METHODS=(lora+pnp lora+directinversion+pnp) ;;
  masactrl) METRIC_METHODS=(lora+masactrl lora+directinversion+masactrl) ;;
  pix2pix_zero) METRIC_METHODS=(lora+pix2pix-zero lora+directinversion+pix2pix-zero) ;;
  *) echo "No metrics mapping for editor: $NAME" >&2; exit 2 ;;
esac
METRICS_PYTHON=${METRICS_PYTHON:-$REPO_DIR/.venv-pnp-metrics/bin/python}
[[ -x "$METRICS_PYTHON" ]] || { echo "Metrics Python not found: $METRICS_PYTHON" >&2; exit 2; }
for METHOD in "${METRIC_METHODS[@]}"; do
  "$PYTHON" -m diff_inversion.eval.pnp_lora_metrics --config-name eval/pnp_lora_metrics \
    data_path="$PIE_BENCH_DATA" generated_root="$OUTPUT_ROOT/$NAME" \
    metrics_root="$OUTPUT_ROOT/metrics" log_root="$OUTPUT_ROOT/logs/metrics" \
    cache_root="$PERSISTENT_MODEL_CACHE/metrics" +python="$METRICS_PYTHON" method_name="$METHOD"
done
