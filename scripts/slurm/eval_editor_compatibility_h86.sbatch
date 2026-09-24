#!/bin/bash
#SBATCH --job-name=editor-compat
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

# Run one paired-control cell as an ordinary Slurm job (never an array).
# Usage:
#   sbatch ... pnp_source base|lora smoke|full
#   sbatch ... pix_normalized base|lora smoke|full 0.1
EXPERIMENT=${1:?Pass pnp_source or pix_normalized}
MODE=${2:?Pass base or lora}
SET=${3:-smoke}
PIX_GUIDANCE=${4:-0.1}
PIX_LOSS_SCALE=${5:-1.0}
PIX_CLIP_NORM=${6:-1.0}

case "$EXPERIMENT" in pnp_source|pix_normalized) ;; *) echo "Invalid experiment: $EXPERIMENT" >&2; exit 2 ;; esac
case "$MODE" in base|lora) ;; *) echo "Invalid mode: $MODE" >&2; exit 2 ;; esac
case "$SET" in smoke|diagnostic20|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
CHECKPOINT_REL=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt
[[ -f "$REPO_DIR/$CHECKPOINT_REL" ]] || { echo "Missing checkpoint: $CHECKPOINT_REL" >&2; exit 2; }

if [[ "$SET" == smoke ]]; then
  DATA_PATH=checkpoints/../evaluation/reconstruction_diagnostics/manifests/smoke
elif [[ "$SET" == diagnostic20 ]]; then
  DATA_PATH=checkpoints/../evaluation/reconstruction_diagnostics/manifests/pix2pix_worst20
else
  DATA_PATH=data/raw/PIE-Bench_v1
fi

if [[ "$EXPERIMENT" == pnp_source ]]; then
  TAG=pnp_source_reference
  BACKEND=pnp
  SUFFIX=pnp
else
  GUIDANCE_TAG=${PIX_GUIDANCE//./_}
  SCALE_TAG=${PIX_LOSS_SCALE//./_}
  CLIP_TAG=${PIX_CLIP_NORM//./_}
  TAG=pix2pix_normalized_g${GUIDANCE_TAG}_s${SCALE_TAG}_clip${CLIP_TAG}
  BACKEND=pix2pix_zero
  SUFFIX=pix2pix-zero
fi

OUTPUT_PATH=checkpoints/../evaluation/editor_compatibility/$SET/$TAG
if [[ "$MODE" == base ]]; then
  METHOD=directinversion+$SUFFIX
else
  METHOD=lora+directinversion+$SUFFIX
fi

JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/editor-compat-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv

module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() {
  [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true
  rm -rf -- "$JOB_TMP_DIR"
}
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"

cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

if [[ "$MODE" == base ]]; then
  if [[ "$EXPERIMENT" == pnp_source ]]; then
    CMD=(
      python3 -m diff_inversion.eval.sd15_baseline
      --config-name eval/sd15_baseline
      backend="$BACKEND"
      inversion=directinversion
      data_path="$DATA_PATH"
      output_path="$OUTPUT_PATH"
      cache_root="$JOB_CACHE_ROOT/editing"
      pnp.reference_prompt_mode=source
    )
    "${CMD[@]}"
  else
    CMD=(
      python3 -m diff_inversion.eval.sd15_baseline
      --config-name eval/sd15_baseline
      backend="$BACKEND"
      inversion=directinversion
      data_path="$DATA_PATH"
      output_path="$OUTPUT_PATH"
      cache_root="$JOB_CACHE_ROOT/editing"
      pix2pix_zero.guidance_amount="$PIX_GUIDANCE"
      pix2pix_zero.attention_loss_reduction=per_layer_mean
      pix2pix_zero.attention_loss_scale="$PIX_LOSS_SCALE"
    )
    if [[ "$PIX_CLIP_NORM" != none ]]; then
      CMD+=(pix2pix_zero.attention_grad_clip_norm="$PIX_CLIP_NORM")
    fi
    "${CMD[@]}"
  fi
else
  if [[ "$EXPERIMENT" == pnp_source ]]; then
    MODULE=diff_inversion.eval.pnp_lora
    CONFIG=eval/pnp_lora
    EXTRA_ARGS=(pnp.reference_prompt_mode=source)
  else
    MODULE=diff_inversion.eval.pix2pix_zero_lora
    CONFIG=eval/pix2pix_zero_lora
    EXTRA_ARGS=(
      pix2pix_zero.guidance_amount="$PIX_GUIDANCE"
      pix2pix_zero.attention_loss_reduction=per_layer_mean
      pix2pix_zero.attention_loss_scale="$PIX_LOSS_SCALE"
    )
    if [[ "$PIX_CLIP_NORM" != none ]]; then
      EXTRA_ARGS+=(pix2pix_zero.attention_grad_clip_norm="$PIX_CLIP_NORM")
    fi
  fi
  CMD=(
    python3 -m "$MODULE"
    --config-name "$CONFIG"
    "edit_method_list=[$METHOD]"
    lora=r16
    lora.checkpoint_path="$CHECKPOINT_REL"
    lora.mode=single
    data_path="$DATA_PATH"
    output_path="$OUTPUT_PATH"
    cache_root="$JOB_CACHE_ROOT/editing"
    guidance.inversion_scale=1.0
  )
  "${CMD[@]}" "${EXTRA_ARGS[@]}"
fi

METRICS_CMD=(
  python3 -m diff_inversion.eval.pnp_lora_metrics
  --config-name eval/pnp_lora_metrics
  "method_name=$METHOD"
  data_path="$DATA_PATH"
  generated_root="$OUTPUT_PATH"
  metrics_root="$OUTPUT_PATH/metrics"
  log_root="$JOB_CACHE_ROOT/metric-logs"
  cache_root="$JOB_CACHE_ROOT/metrics"
)
"${METRICS_CMD[@]}"

echo "Completed experiment=$EXPERIMENT mode=$MODE set=$SET guidance=$PIX_GUIDANCE loss_scale=$PIX_LOSS_SCALE clip=$PIX_CLIP_NORM"
