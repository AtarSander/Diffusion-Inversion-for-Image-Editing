#!/bin/bash
#SBATCH --job-name=sd15-cycle-invqual
#SBATCH --partition=gpu-m
#SBATCH --nodelist=h32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

VARIANT=${1:?Usage: sbatch eval_sd15_cycle_inversion_quality_h32.sbatch VARIANT RANK CHECKPOINT_OR_NONE}
RANK=${2:?Usage: sbatch eval_sd15_cycle_inversion_quality_h32.sbatch VARIANT RANK CHECKPOINT_OR_NONE}
CHECKPOINT=${3:?Usage: sbatch eval_sd15_cycle_inversion_quality_h32.sbatch VARIANT RANK CHECKPOINT_OR_NONE}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-1522}

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
SOURCE_ROOT=$REPO_DIR/data/processed/sd15_trajectories_stacked_fp32/all
MANIFEST=$REPO_DIR/reports/eval/sd15_cycle_inversion_quality/manifest.jsonl
OUTPUT_DIR=$REPO_DIR/reports/eval/sd15_cycle_inversion_quality/$VARIANT
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/sd15-cycle-invqual-${SLURM_JOB_ID}
VIEW_ROOT=$JOB_TMP_DIR/view
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$OUTPUT_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m scripts.analysis.prepare_sd15_inversion_quality_view \
  --source-root "$SOURCE_ROOT" \
  --manifest "$MANIFEST" \
  --expected-samples "$EXPECTED_SAMPLES" \
  --max-samples "$EXPECTED_SAMPLES" \
  --view-root "$VIEW_ROOT"

LORA_ARGS=(lora.enabled=false)
if [[ "$CHECKPOINT" != "none" ]]; then
  [[ -f "$REPO_DIR/$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
  LORA_ARGS=(
    lora.enabled=true
    lora.checkpoint_path="$CHECKPOINT"
    lora.r="$RANK"
    lora.lora_alpha="$((RANK / 2))"
  )
fi

python3 -m diff_inversion.eval.invert_sdxl --config-name eval/invert_sdxl \
  input_dir="$VIEW_ROOT" overwrite=true save_intermediates=false "${LORA_ARGS[@]}"

# Reconstruction must use the frozen base model, without the inversion adapter.
python3 -m diff_inversion.eval.reconstruct_sdxl --config-name eval/reconstruct_sdxl \
  input_dir="$VIEW_ROOT" overwrite=true lora.enabled=false

python3 -m diff_inversion.eval.run --config-name eval/run \
  input_dir="$VIEW_ROOT" output_dir="$OUTPUT_DIR" max_samples="$EXPECTED_SAMPLES" \
  save_noise_previews=false save_normality_plots=false \
  inversion_diagnostics.enabled=false calculate_lpips=true lpips_device=cuda

python3 -m scripts.analysis.summarize_sd15_cycle_inversion_quality
