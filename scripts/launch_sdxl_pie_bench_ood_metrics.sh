#!/usr/bin/env bash
# Calculate aggregate inversion-quality metrics from completed PIE-Bench OOD outputs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-04:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-48G}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_pie_bench_ood_metrics}
WORK_ROOT=${WORK_ROOT:-$REPO_DIR/data/processed/sdxl_trajectories_pie_bench_ood}
REPORT_ROOT=${REPORT_ROOT:-$REPO_DIR/reports/eval/sdxl_pie_bench_ood}
TOTAL_PROMPTS=${TOTAL_PROMPTS:-700}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  JOB_ID=$(sbatch --parsable --job-name=sdxl-pie-ood-metrics --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --gres="$SLURM_GRES" \
    --time="$SLURM_TIME" --array=0-3 \
    --output="$RUN_LOG_DIR/metrics_%A_%a.out" --error="$RUN_LOG_DIR/metrics_%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",WORK_ROOT="$WORK_ROOT",REPORT_ROOT="$REPORT_ROOT",TOTAL_PROMPTS="$TOTAL_PROMPTS" \
    "$0" --worker)
  echo "Submitted metrics-only array: $JOB_ID"
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$SLURM_ARRAY_TASK_ID" in
  0) METHOD=ddim ;;
  1) METHOD=lora_r8 ;;
  2) METHOD=lora_r16 ;;
  3) METHOD=lora_r32 ;;
  *) echo "Unexpected metrics task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-pie-ood-metrics-${SLURM_JOB_ID}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"

INPUT_DIR="$WORK_ROOT/$METHOD"
[[ -d $INPUT_DIR ]] || { echo "Completed output directory not found: $INPUT_DIR" >&2; exit 2; }
"$PYTHON" -m diff_inversion.eval.run input_dir="$INPUT_DIR" output_dir="$REPORT_ROOT/$METHOD" \
  max_samples="$TOTAL_PROMPTS" wandb.mode=disabled
