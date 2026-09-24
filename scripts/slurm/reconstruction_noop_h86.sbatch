#!/bin/bash
#SBATCH --job-name=recon-noop
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
MODE=${1:?Pass base or lora}
EDITOR=${2:?Pass p2p, pnp, masactrl, or pix2pix_zero}
SET=${3:-full}
case "$MODE" in base|lora) ;; *) echo "Invalid mode: $MODE" >&2; exit 2 ;; esac
case "$EDITOR" in p2p|pnp|masactrl|pix2pix_zero) ;; *) echo "Invalid editor: $EDITOR" >&2; exit 2 ;; esac
case "$SET" in smoke|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac
REPO_DIR=${SLURM_SUBMIT_DIR:?}
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
DIAG_ROOT=$STORAGE_ROOT/evaluation/reconstruction_diagnostics
DATA_PATH=$DIAG_ROOT/manifests/$([[ "$SET" == smoke ]] && echo smoke_noop || echo noop)
OUTPUT_PATH=$DIAG_ROOT/$SET/noop
CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/recon-noop-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR" "$OUTPUT_PATH"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev
if [[ "$MODE" == base ]]; then
  python3 -m diff_inversion.eval.sd15_baseline --config-name eval/sd15_baseline \
    backend="$EDITOR" inversion=directinversion data_path="$DATA_PATH" \
    output_path="$OUTPUT_PATH" cache_root="$JOB_CACHE_ROOT/editing"
else
  case "$EDITOR" in
    p2p) MODULE=diff_inversion.eval.p2p_lora; CONFIG=eval/p2p_lora; SUFFIX=p2p ;;
    pnp) MODULE=diff_inversion.eval.pnp_lora; CONFIG=eval/pnp_lora; SUFFIX=pnp ;;
    masactrl) MODULE=diff_inversion.eval.masactrl_lora; CONFIG=eval/masactrl_lora; SUFFIX=masactrl ;;
    pix2pix_zero) MODULE=diff_inversion.eval.pix2pix_zero_lora; CONFIG=eval/pix2pix_zero_lora; SUFFIX=pix2pix-zero ;;
  esac
  python3 -m "$MODULE" --config-name "$CONFIG" \
    "edit_method_list=[lora+directinversion+$SUFFIX]" lora=r16 \
    lora.checkpoint_path="$REPO_DIR/$CHECKPOINT_REL" lora.mode=single \
    data_path="$DATA_PATH" output_path="$OUTPUT_PATH" \
    cache_root="$JOB_CACHE_ROOT/editing" guidance.inversion_scale=1.0
fi
