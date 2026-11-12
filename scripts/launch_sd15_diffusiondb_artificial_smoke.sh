#!/usr/bin/env bash
# One guarded GPU smoke: generate two artificial trajectories, then train two mixture updates.
set -euo pipefail
[[ ${DEBUG_SHELL:-0} == 1 ]] && set -x
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgdiffusion3-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
SLURM_GRES=${SLURM_GRES:-gpu:1}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}
SLURM_MEM=${SLURM_MEM:-128G}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sd15_diffusiondb_artificial_smoke}
SMOKE_ACK_FILE=${SMOKE_ACK_FILE:-$REPO_DIR/data/processed/diffusiondb_artificial/smoke_acknowledgement.json}

if [[ ${1:-} != --worker ]]; then
  [[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=sd15-diffdb-smoke --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" \
    --gres="$SLURM_GRES" --time="$SLURM_TIME" --output="$RUN_LOG_DIR/%j.out" --error="$RUN_LOG_DIR/%j.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON" "$0" --worker
fi

# Athena does not export SLURM_TMPDIR; /tmp is node-local on its compute nodes.
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/slurm-${SLURM_JOB_ID}}"
JOB_TMP_DIR="$SLURM_TMPDIR/sd15-diffdb-smoke-${SLURM_JOB_ID}"
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
mkdir -p "$JOB_TMP_DIR"
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
source "$REPO_DIR/scripts/slurm/stage_sd15_model.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
MODEL_DIR="$JOB_TMP_DIR/models/stable-diffusion-v1-5"
stage_sd15_model "$MODEL_DIR"
"$PYTHON" -m diff_inversion.data.generate_sdxl_samples \
  --config-name sample_gather_sd15_diffusiondb_artificial_smoke model.model_id="$MODEL_DIR" model.disable_safety_checker=true
"$PYTHON" -m diff_inversion.modeling.train \
  --config-name train_sd15_diffusiondb_artificial_mixture_smoke model.model_id="$MODEL_DIR" model.disable_safety_checker=true wandb.mode=disabled
"$PYTHON" - "$SMOKE_ACK_FILE" <<'SMOKE_PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"completed": True, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}) + "\n", encoding="utf-8")
SMOKE_PY
