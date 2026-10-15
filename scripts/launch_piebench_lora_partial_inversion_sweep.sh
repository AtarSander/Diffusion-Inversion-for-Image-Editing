#!/usr/bin/env bash
# Each array task runs four LoRA editor pairs and their metrics.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
METRICS_PYTHON=${METRICS_PYTHON:-$REPO_DIR/.venv-pnp-metrics/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
LOG_ROOT=${LOG_ROOT:-$REPO_DIR/data/logs/slurm/piebench_lora_partial_inversion}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_DIR/data/evaluation/pnp_lora_partial_inversion_r16}
CHECKPOINT=${CHECKPOINT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/checkpoints/sd15_inversion_lora_r16_lr5e-5_cosine_bf16/checkpoint_step_130000.pt}
PIE_BENCH_DATA=${PIE_BENCH_DATA:-$REPO_DIR/data/raw/PIE-Bench_v1}
MODEL_CACHE=${MODEL_CACHE:-$REPO_DIR/data/model_cache/piebench}
SLURM_ARRAY=${SLURM_ARRAY:-10-90:10}

if [[ ${1:-} != --worker ]]; then
  [[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
  [[ -f "$PIE_BENCH_DATA/mapping_file.json" ]] || { echo "Missing PIE-Bench mapping" >&2; exit 2; }
  [[ -x "$PYTHON" && -x "$METRICS_PYTHON" ]] || { echo "Missing Python environment" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=piebench-lora-partial --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=24 \
    --mem=128G --gres=gpu:4 --time=48:00:00 --array="$SLURM_ARRAY" \
    --output="$RUN_LOG_DIR/%A_%a.out" --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",METRICS_PYTHON="$METRICS_PYTHON",OUTPUT_ROOT="$OUTPUT_ROOT",CHECKPOINT="$CHECKPOINT",PIE_BENCH_DATA="$PIE_BENCH_DATA",MODEL_CACHE="$MODEL_CACHE" \
    "$0" --worker
fi

PERCENT=${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}
(( PERCENT >= 10 && PERCENT <= 100 && PERCENT % 2 == 0 )) || {
  echo "Invalid sweep percentage: $PERCENT (use an even percentage from 10 to 100 for 50 inversion steps)" >&2
  exit 2
}
INVERSION_STEPS=$((PERCENT / 2))
RUN_ROOT="$OUTPUT_ROOT/p${PERCENT}"
mkdir -p "$RUN_ROOT/logs"
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_ARRAY_JOB_ID}-${PERCENT}}"
JOB_TMP_DIR="$SLURM_TMPDIR/piebench-lora-partial"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
cd "$REPO_DIR"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#GPU_IDS[@]} >= 4 )) || { echo "Need four allocated GPUs" >&2; exit 2; }

run_editor() {
  local editor=$1 module=$2 config=$3 first_method=$4 second_method=$5 gpu=$6
  export CUDA_VISIBLE_DEVICES="$gpu"
  local editor_log="$RUN_ROOT/logs/${editor}.log"
  {
    echo "editor=$editor inversion_steps=$INVERSION_STEPS gpu=$gpu"
    "$PYTHON" -m "$module" --config-name "$config" \
      lora=r16 lora.checkpoint_path="$CHECKPOINT" \
      data_path="$PIE_BENCH_DATA" output_path="$RUN_ROOT" \
      cache_root="$MODEL_CACHE/$editor" \
      edit_method_list="[$first_method,$second_method]" \
      inversion_steps="$INVERSION_STEPS" \
      hydra.run.dir="$RUN_ROOT/logs/hydra/$editor"

    for method in "$first_method" "$second_method"; do
      "$PYTHON" -m diff_inversion.eval.pnp_lora_metrics --config-name eval/pnp_lora_metrics \
        data_path="$PIE_BENCH_DATA" generated_root="$RUN_ROOT" \
        metrics_root="$RUN_ROOT/metrics" log_root="$RUN_ROOT/logs/metrics" \
        cache_root="$MODEL_CACHE/metrics" +python="$METRICS_PYTHON" method_name="$method"
    done
  } >"$editor_log" 2>&1
}

run_editor p2p diff_inversion.eval.p2p_lora eval/p2p_lora \
  lora+p2p lora+directinversion+p2p "${GPU_IDS[0]}" & pids=("$!")
run_editor pnp diff_inversion.eval.pnp_lora eval/pnp_lora \
  lora+pnp lora+directinversion+pnp "${GPU_IDS[1]}" & pids+=("$!")
run_editor masactrl diff_inversion.eval.masactrl_lora eval/masactrl_lora \
  lora+masactrl lora+directinversion+masactrl "${GPU_IDS[2]}" & pids+=("$!")
run_editor pix2pix_zero diff_inversion.eval.pix2pix_zero_lora eval/pix2pix_zero_lora \
  lora+pix2pix-zero lora+directinversion+pix2pix-zero "${GPU_IDS[3]}" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
