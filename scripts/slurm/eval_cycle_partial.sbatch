#!/bin/bash
#SBATCH --job-name=cycle-partial
#SBATCH --partition=gpu-m,gpu-l
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

PERCENT=${1:?Pass 94, 96, or 98}
EDITOR=${2:?Pass p2p, pnp, masactrl, or pix2pix_zero}
case "$PERCENT" in 94|96|98) ;; *) exit 2 ;; esac
case "$EDITOR" in p2p|pnp|masactrl|pix2pix_zero) ;; *) exit 2 ;; esac
STEPS=$((PERCENT / 2))
METHOD="lora+directinversion+${EDITOR//_/-}"
REPO_DIR=${SLURM_SUBMIT_DIR:?}
cd "$REPO_DIR"
module load python/3.11.15 uv/0.11.2
JOB_TMP_DIR=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/cycle-partial-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$JOB_TMP_DIR"' EXIT
source scripts/slurm/job_local_cache.sh
export UV_PROJECT_ENVIRONMENT="$JOB_TMP_DIR/venv"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
uv sync --frozen --no-dev

# Resolve persistent paths from Hydra, not machine-specific launcher defaults.
RESOLVED=$(uv run --no-sync python - "$STEPS" <<'PY'
import sys
from pathlib import Path
from hydra import initialize_config_dir, compose
with initialize_config_dir(config_dir=str(Path('config').resolve()), version_base=None):
    cfg = compose(config_name='eval/cycle_partial', overrides=[f'inversion_steps={sys.argv[1]}'])
    assert Path(cfg.lora.checkpoint_path).is_file(), cfg.lora.checkpoint_path
    assert (Path(cfg.data_path) / 'mapping_file.json').is_file(), cfg.data_path
    print(cfg.output_path)
    print(cfg.data_path)
    print(f'checkpoint={cfg.lora.checkpoint_path}', file=sys.stderr)
PY
)
mapfile -t PATHS <<< "$RESOLVED"
OUTPUT_ROOT=${PATHS[0]}
DATA_PATH=${PATHS[1]}
echo "editor=$EDITOR percent=$PERCENT steps=$STEPS method=$METHOD output=$OUTPUT_ROOT"

uv run --no-sync python -m "diff_inversion.eval.${EDITOR}_lora" \
  --config-name eval/cycle_partial "edit_method_list=[$METHOD]" inversion_steps="$STEPS"

uv run --no-sync python -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics "method_name=$METHOD" \
  data_path="$DATA_PATH" generated_root="$OUTPUT_ROOT" \
  metrics_root="$OUTPUT_ROOT/metrics" log_root="$OUTPUT_ROOT/logs" \
  cache_root="$JOB_CACHE_ROOT/metrics"
