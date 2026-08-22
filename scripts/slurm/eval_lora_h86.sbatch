#!/bin/bash
#SBATCH --job-name=lora-edit
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=0-7%7
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

RANK=${1:?Pass rank 8, 16, or 32}
INVERSION_GUIDANCE=${2:-1.0}
case "$RANK" in
  8) CHECKPOINT_STEP=130000 ;;
  16) CHECKPOINT_STEP=130000 ;;
  32) CHECKPOINT_STEP=125000 ;;
  *) echo "Unsupported rank: $RANK (expected 8, 16, or 32)" >&2; exit 2 ;;
esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
STORAGE_ROOT="$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")"
[[ -d "$STORAGE_ROOT" ]] || { echo "Cannot resolve project storage from $REPO_DIR/checkpoints" >&2; exit 2; }
BASE=$STORAGE_ROOT
CHECKPOINT=$BASE/checkpoints/r$RANK/checkpoint_step_${CHECKPOINT_STEP}.pt
OUTPUT_TAG=r$RANK
if [[ "$INVERSION_GUIDANCE" != "1" && "$INVERSION_GUIDANCE" != "1.0" ]]; then
  OUTPUT_TAG=inv_guidance_${INVERSION_GUIDANCE//./_}/r$RANK
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODULES=(
  diff_inversion.eval.p2p_lora diff_inversion.eval.p2p_lora
  diff_inversion.eval.pnp_lora diff_inversion.eval.pnp_lora
  diff_inversion.eval.masactrl_lora diff_inversion.eval.masactrl_lora
  diff_inversion.eval.pix2pix_zero_lora diff_inversion.eval.pix2pix_zero_lora
)
CONFIGS=(
  eval/p2p_lora eval/p2p_lora
  eval/pnp_lora eval/pnp_lora
  eval/masactrl_lora eval/masactrl_lora
  eval/pix2pix_zero_lora eval/pix2pix_zero_lora
)
METHODS=(
  lora+p2p lora+directinversion+p2p
  lora+pnp lora+directinversion+pnp
  lora+masactrl lora+directinversion+masactrl
  lora+pix2pix-zero lora+directinversion+pix2pix-zero
)
MODULE=${MODULES[$TASK_ID]:-}
CONFIG=${CONFIGS[$TASK_ID]:-}
METHOD=${METHODS[$TASK_ID]:-}
[[ -n "$METHOD" ]] || { echo "Invalid array task: $TASK_ID" >&2; exit 2; }

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/lora-edit-${SLURM_ARRAY_JOB_ID}-$TASK_ID
VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m "$MODULE" --config-name "$CONFIG" \
  "edit_method_list=[$METHOD]" \
  lora="r$RANK" \
  lora.checkpoint_path="$CHECKPOINT" \
  output_path="$BASE/evaluation/$OUTPUT_TAG" \
  cache_root="$JOB_CACHE_ROOT/editing" \
  guidance.inversion_scale="$INVERSION_GUIDANCE"
