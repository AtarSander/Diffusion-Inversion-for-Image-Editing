#!/usr/bin/env bash
# Submit only the controlled r16 80/20 FP32 mixture run after prompt review and trajectory generation.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-2-00:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}
SLURM_MEM=${SLURM_MEM:-32G}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sd15_diffusiondb_artificial_training}
REVIEW_ACK_FILE=${REVIEW_ACK_FILE:-$REPO_DIR/data/processed/diffusiondb_artificial/review_acknowledgement.json}
SMOKE_ACK_FILE=${SMOKE_ACK_FILE:-$REPO_DIR/data/processed/diffusiondb_artificial/smoke_acknowledgement.json}

check_review() {
  [[ -f $REVIEW_ACK_FILE ]] || { echo "Missing review acknowledgement: $REVIEW_ACK_FILE" >&2; exit 2; }
  "$PYTHON" - "$REVIEW_ACK_FILE" <<'REVIEW_PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    acknowledgement = json.load(handle)
if acknowledgement.get("reviewed") is not True:
    raise SystemExit("review_acknowledgement.json must contain {\"reviewed\": true}.")
REVIEW_PY
}

check_smoke() {
  [[ -f $SMOKE_ACK_FILE ]] || { echo "GPU smoke acknowledgement is required: $SMOKE_ACK_FILE" >&2; exit 2; }
  "$PYTHON" - "$SMOKE_ACK_FILE" <<'SMOKE_PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    acknowledgement = json.load(handle)
if acknowledgement.get("completed") is not True:
    raise SystemExit("smoke_acknowledgement.json must contain {\"completed\": true}.")
SMOKE_PY
}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  check_review
  check_smoke
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=sd15-r16-recap80-diffdb20 --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" \
    --gres="$SLURM_GRES" --time="$SLURM_TIME" --output="$RUN_LOG_DIR/%j.out" --error="$RUN_LOG_DIR/%j.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON" "$0" --worker "$@"
fi

if [[ ${1:-} == --worker ]]; then shift; fi
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sd15-r16-recap80-diffdb20-${SLURM_JOB_ID}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
"$PYTHON" -m diff_inversion.modeling.train --config-name train_sd15_diffusiondb_artificial_mixture "$@"
