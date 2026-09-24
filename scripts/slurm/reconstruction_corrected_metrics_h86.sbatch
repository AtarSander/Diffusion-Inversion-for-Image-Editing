#!/bin/bash
#SBATCH --job-name=recon-corrected
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
SCOPE=${1:?Pass existing, noop, masactrl3, or pixcross}
EDITOR=${2:-none}
SET=${3:-full}
case "$SET" in smoke|full) ;; *) echo "Invalid set: $SET" >&2; exit 2 ;; esac
case "$SCOPE" in existing|noop|masactrl3|pixcross) ;; *) echo "Invalid scope: $SCOPE" >&2; exit 2 ;; esac
REPO_DIR=${SLURM_SUBMIT_DIR:?}
STORAGE_ROOT=$(dirname "$(readlink -f "$REPO_DIR/checkpoints")")
DIAG_ROOT=$STORAGE_ROOT/evaluation/reconstruction_diagnostics
JOB_TMP_DIR=${SLURM_TMPDIR:-/tmp}/recon-corrected-${SLURM_JOB_ID}
VENV_DIR=$JOB_TMP_DIR/venv
module load python/3.11.15
module load uv/0.11.2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source "$REPO_DIR/scripts/slurm/job_local_cache.sh"
cleanup() { [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; rm -rf -- "$JOB_TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$JOB_TMP_DIR"
cd "$REPO_DIR"
uv venv --python "$(command -v python3)" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync --active --frozen --no-dev

run_corrected() {
  local mapping=$1 source=$2 output=$3 summary=$4
  shift 4
  python3 -m diff_inversion.eval.corrected_reconstruction_metrics \
    --mapping-file "$mapping" --source-root "$source" \
    --output-csv "$output" --summary-json "$summary" --device cuda "$@"
}

if [[ "$SCOPE" == existing ]]; then
  case "$EDITOR" in p2p|pnp|masactrl|pix2pix-zero) ;; *) echo "Invalid editor: $EDITOR" >&2; exit 2 ;; esac
  OUTPUT_DIR=$DIAG_ROOT/full/corrected_existing/$EDITOR
  mkdir -p "$OUTPUT_DIR"
  run_corrected "$REPO_DIR/data/raw/PIE-Bench_v1/mapping_file.json" \
    "$REPO_DIR/data/raw/PIE-Bench_v1/annotation_images" \
    "$OUTPUT_DIR/per_image.csv" "$OUTPUT_DIR/summary.json" \
    --method "directinv=$STORAGE_ROOT/evaluation/sd15_fp16_baselines/directinversion+$EDITOR/annotation_images" \
    --method "lora_directinv=$STORAGE_ROOT/evaluation/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors/lora+directinversion+$EDITOR/annotation_images"
elif [[ "$SCOPE" == noop ]]; then
  case "$EDITOR" in p2p|pnp|masactrl|pix2pix-zero) ;; *) echo "Invalid editor: $EDITOR" >&2; exit 2 ;; esac
  OUTPUT_DIR=$DIAG_ROOT/$SET/noop_metrics/$EDITOR
  mkdir -p "$OUTPUT_DIR"
  COMMON=(
    --method "directinv=$DIAG_ROOT/$SET/noop/directinversion+$EDITOR/annotation_images"
    --method "lora_directinv=$DIAG_ROOT/$SET/noop/lora+directinversion+$EDITOR/annotation_images"
    --full-image
  )
  NOOP_DATA=$DIAG_ROOT/manifests/$([[ "$SET" == smoke ]] && echo smoke_noop || echo noop)
  run_corrected "$NOOP_DATA/mapping_file.json" \
    "$NOOP_DATA/annotation_images" \
    "$OUTPUT_DIR/edited_per_image.csv" "$OUTPUT_DIR/edited_summary.json" \
    --panel-index -1 "${COMMON[@]}"
  NOOP_DATA=$DIAG_ROOT/manifests/$([[ "$SET" == smoke ]] && echo smoke_noop || echo noop)
  run_corrected "$NOOP_DATA/mapping_file.json" \
    "$NOOP_DATA/annotation_images" \
    "$OUTPUT_DIR/roundtrip_per_image.csv" "$OUTPUT_DIR/roundtrip_summary.json" \
    --panel-index 2 "${COMMON[@]}"
elif [[ "$SCOPE" == masactrl3 ]]; then
  OUTPUT_DIR=$DIAG_ROOT/$SET/masactrl_conditioning_metrics
  mkdir -p "$OUTPUT_DIR"
  MASA_DATA=$([[ "$SET" == smoke ]] && echo "$DIAG_ROOT/manifests/smoke" || echo "$REPO_DIR/data/raw/PIE-Bench_v1")
  run_corrected "$MASA_DATA/mapping_file.json" \
    "$MASA_DATA/annotation_images" \
    "$OUTPUT_DIR/per_image.csv" "$OUTPUT_DIR/summary.json" \
    --method "base_empty=$STORAGE_ROOT/evaluation/sd15_fp16_baselines/directinversion+masactrl/annotation_images" \
    --method "base_source=$DIAG_ROOT/$SET/masactrl_source_prompt/directinversion+masactrl/annotation_images" \
    --method "lora_source=$STORAGE_ROOT/evaluation/one_epoch/cycle_loss_cfg1_r16_uniform_e1_all_editors/lora+directinversion+masactrl/annotation_images"
else
  OUTPUT_DIR=$DIAG_ROOT/$SET/pix2pix_crossing_metrics
  DATA_ROOT=$DIAG_ROOT/manifests/$([[ "$SET" == smoke ]] && echo smoke || echo pix2pix_worst20)
  IMAGE_ROOT=$DIAG_ROOT/$SET/pix2pix_crossing
  mkdir -p "$OUTPUT_DIR"
  METHOD_ARGS=()
  TARGET_ARGS=()
  for endpoint in base lora; do
    for attention in base lora; do
      for offsets in base lora; do
        name=pix2pix-cross_e-${endpoint}_a-${attention}_o-${offsets}
        directory=$IMAGE_ROOT/$name/annotation_images
        METHOD_ARGS+=(--method "$name=$directory")
        TARGET_ARGS+=(--tgt_image_folder "$name=$directory")
      done
    done
  done
  run_corrected "$DATA_ROOT/mapping_file.json" "$DATA_ROOT/annotation_images" \
    "$OUTPUT_DIR/corrected_per_image.csv" "$OUTPUT_DIR/corrected_summary.json" \
    "${METHOD_ARGS[@]}"
  cd "$REPO_DIR/pnp_inversion"
  python3 -m evaluation.evaluate \
    --annotation_mapping_file "$DATA_ROOT/mapping_file.json" \
    --src_image_folder "$DATA_ROOT/annotation_images" \
    "${TARGET_ARGS[@]}" --result_path "$OUTPUT_DIR/official_metrics.csv" --device cuda \
    --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part \
      ssim_unedit_part clip_similarity_source_image clip_similarity_target_image \
      clip_similarity_target_image_edit_part \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9
fi
