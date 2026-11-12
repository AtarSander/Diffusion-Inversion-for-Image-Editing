#!/usr/bin/env bash
# Evaluate the SDXL r8/r16/r32 LoRA one-transition cycle metric on Athena.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-06:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-48G}
MAX_BATCHES=${MAX_BATCHES:-128}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_cycle_metrics}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/checkpoints}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=sdxl-cycle --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" \
    --gres="$SLURM_GRES" --time="$SLURM_TIME" --array=0-2 \
    --output="$RUN_LOG_DIR/%A_%a.out" --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",MAX_BATCHES="$MAX_BATCHES" \
    "$0" --worker
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
case "$SLURM_ARRAY_TASK_ID" in
  0) RANK=8; ALPHA=4 ;;
  1) RANK=16; ALPHA=8 ;;
  2) RANK=32; ALPHA=16 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-cycle-r${RANK}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

CHECKPOINT_PATH="$CHECKPOINT_ROOT/sdxl_inversion_lora_r${RANK}_lr5e-5_cosine_bf16/checkpoint_step_15000.pt"
OUTPUT_PATH="$REPO_DIR/reports/sdxl_inversion_lora_r${RANK}_step15000_cycle_metrics.json"
[[ -f $CHECKPOINT_PATH ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 2; }

cd "$REPO_DIR"
"$PYTHON" -m diff_inversion.eval.sdxl_cycle_metrics \
  lora="r${RANK}" \
  lora.r="$RANK" \
  lora.lora_alpha="$ALPHA" \
  checkpoint_path="$CHECKPOINT_PATH" \
  output_path="$OUTPUT_PATH" \
  max_batches="$MAX_BATCHES"
