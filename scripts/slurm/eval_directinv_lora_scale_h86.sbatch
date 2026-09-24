#!/bin/bash
# Adapter-scale sweep for existing best standard-CFG1 LoRA+Direct-Inversion
# adapters.  This generates edited PIE panels and metrics; scale 0 is the
# existing Direct-Inversion control and scale 1 is already available.
#
# Usage: sbatch scripts/slurm/eval_directinv_lora_scale_h86.sbatch EDITOR SCALE
# EDITOR: p2p | pnp | masactrl | pix2pix-zero
# SCALE:  0.25 | 0.5 | 0.75
#SBATCH --job-name=directinv-lora-scale
#SBATCH --partition=gpu-l
#SBATCH --nodelist=h86
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
EDITOR=${1:?Pass p2p, pnp, masactrl, or pix2pix-zero}
SCALE=${2:?Pass 0.25, 0.5, or 0.75}
case "$SCALE" in 0.25|0.5|0.75) ;; *) echo "Unsupported scale: $SCALE" >&2; exit 2 ;; esac

case "$EDITOR" in
  p2p)
    MODULE=diff_inversion.eval.p2p_lora
    CONFIG=eval/p2p_lora
    METHOD=lora+directinversion+p2p
    CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r8_bs128_e1/checkpoint_final.pt
    RANK=8
    ;;
  pnp)
    MODULE=diff_inversion.eval.pnp_lora
    CONFIG=eval/pnp_lora
    METHOD=lora+directinversion+pnp
    CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r8_bs128_e1/checkpoint_final.pt
    RANK=8
    ;;
  masactrl)
    MODULE=diff_inversion.eval.masactrl_lora
    CONFIG=eval/masactrl_lora
    METHOD=lora+directinversion+masactrl
    CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r16_uniform_e1/checkpoint_final.pt
    RANK=16
    ;;
  pix2pix-zero)
    MODULE=diff_inversion.eval.pix2pix_zero_lora
    CONFIG=eval/pix2pix_zero_lora
    METHOD=lora+directinversion+pix2pix-zero
    CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r16_uniform_e1/checkpoint_final.pt
    RANK=16
    ;;
  *) echo "Invalid editor: $EDITOR" >&2; exit 2 ;;
esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT=$REPO_DIR/$CHECKPOINT_REL
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
SCALE_TAG=${SCALE/./_}
EVAL_ROOT=$STORAGE_ROOT/evaluation/directinv_lora_scale_sweep/cfg1/scale_$SCALE_TAG
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/directinv-lora-scale-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$EVAL_ROOT"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

python3 -m "$MODULE" --config-name "$CONFIG" \
  "edit_method_list=[$METHOD]" \
  lora="r$RANK" \
  lora.checkpoint_path="$CHECKPOINT" \
  lora.mode=single \
  lora.scale="$SCALE" \
  output_path="$EVAL_ROOT" \
  cache_root="$JOB_CACHE_ROOT/editing" \
  guidance.inversion_scale=1.0

python3 -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval/pnp_lora_metrics \
  "method_name=$METHOD" \
  generated_root="$EVAL_ROOT" \
  metrics_root="$EVAL_ROOT/metrics" \
  log_root="$EVAL_ROOT/logs" \
  cache_root="$JOB_CACHE_ROOT/metrics"
