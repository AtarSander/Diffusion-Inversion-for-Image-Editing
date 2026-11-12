#!/usr/bin/env bash
# Find the largest SDXL conditional-cycle micro-batch that fits on one Athena GPU.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MEM=${SLURM_MEM:-64G}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}
NUM_UPDATES=${NUM_UPDATES:-2}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sdxl_conditional_cycle_batch_stress}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=sdxl-cycle-cap --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --gres="$SLURM_GRES" \
    --time="$SLURM_TIME" --output="$RUN_LOG_DIR/%j.out" --error="$RUN_LOG_DIR/%j.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",MAX_BATCH_SIZE="$MAX_BATCH_SIZE",NUM_UPDATES="$NUM_UPDATES" \
    "$0" --worker
fi

export SLURM_TMPDIR="${SLURM_TMPDIR:-/dev/shm/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sdxl-cycle-cap"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sdxl_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"
MODEL_DIR="/dev/shm/${USER:-unknown}/sdxl-shared-model/stable-diffusion-xl-base-1.0"
stage_sdxl_model "$MODEL_DIR"

"$PYTHON" -m diff_inversion.modeling.batch_size_stress_test --config-name train_sdxl_conditional_cycle \
  model.model_id="$MODEL_DIR" lora=r16 checkpoint_dir="$JOB_TMP_DIR/checkpoints" \
  +stress_test.synthetic=false +stress_test.num_updates="$NUM_UPDATES" \
  +stress_test.max_batch_size="$MAX_BATCH_SIZE"
