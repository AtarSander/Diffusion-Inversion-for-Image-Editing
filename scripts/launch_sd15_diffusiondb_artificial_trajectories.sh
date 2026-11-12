#!/usr/bin/env bash
# Current-cluster FP32 artificial trajectory launcher. All scheduler settings are overrideable.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-13:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}
SLURM_MEM=${SLURM_MEM:-32G}
ARRAY_PARALLELISM=${ARRAY_PARALLELISM:-1}
CHUNK_SIZE=${CHUNK_SIZE:-250}
TOTAL_PROMPTS=${TOTAL_PROMPTS:-7610}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sd15_diffusiondb_artificial_trajectories}
REVIEW_ACK_FILE=${REVIEW_ACK_FILE:-$REPO_DIR/data/processed/diffusiondb_artificial/review_acknowledgement.json}
SMOKE_ACK_FILE=${SMOKE_ACK_FILE:-$REPO_DIR/data/processed/diffusiondb_artificial/smoke_acknowledgement.json}

check_review() {
  [[ -f $REVIEW_ACK_FILE ]] || { echo "Manual review acknowledgement is required: $REVIEW_ACK_FILE" >&2; exit 2; }
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
  TASKS=$(((TOTAL_PROMPTS + CHUNK_SIZE - 1) / CHUNK_SIZE))
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=sd15-diffusiondb-traj --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" \
    --gres="$SLURM_GRES" --time="$SLURM_TIME" --array="0-$((TASKS - 1))%${ARRAY_PARALLELISM}" \
    --output="$RUN_LOG_DIR/%A_%a.out" --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",CHUNK_SIZE="$CHUNK_SIZE",TOTAL_PROMPTS="$TOTAL_PROMPTS" \
    "$0" --worker
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
# Athena does not export SLURM_TMPDIR; /tmp is node-local on its compute nodes.
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sd15-diffusiondb-artificial"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sd15_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
(( START_INDEX < TOTAL_PROMPTS )) || exit 0
NUM_SAMPLES=$CHUNK_SIZE
(( START_INDEX + NUM_SAMPLES <= TOTAL_PROMPTS )) || NUM_SAMPLES=$((TOTAL_PROMPTS - START_INDEX))
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
MODEL_DIR="/tmp/${USER:-unknown}/sd15-shared-model/stable-diffusion-v1-5"
stage_sd15_model "$MODEL_DIR"
echo "Generating FP32 artificial trajectories [$START_INDEX:$((START_INDEX + NUM_SAMPLES)))"
"$PYTHON" -m diff_inversion.data.generate_sdxl_samples \
  --config-name sample_gather_sd15_diffusiondb_artificial_fp32 \
  model.model_id="$MODEL_DIR" model.disable_safety_checker=true start_index="$START_INDEX" num_samples="$NUM_SAMPLES"
