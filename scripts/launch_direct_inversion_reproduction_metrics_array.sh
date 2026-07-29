#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv-pnp-metrics/bin/python}
DATA_PATH=${DATA_PATH:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/data/raw/PIE-Bench_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/direct_inversion_reproduction_outputs}
LOG_ROOT=${LOG_ROOT:-/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/logs/direct_inversion_reproduction_metrics}
ARRAY_PARALLELISM=${ARRAY_PARALLELISM:-8}

if [[ ${1:-} != "--worker" ]]; then
  [[ -x $PYTHON ]] || { echo "Metrics Python not found: $PYTHON" >&2; exit 1; }
  [[ -d $DATA_PATH ]] || { echo "PIE-Bench data not found: $DATA_PATH" >&2; exit 1; }
  [[ -d $OUTPUT_ROOT ]] || { echo "Output root not found: $OUTPUT_ROOT" >&2; exit 1; }

  METHODS=()
  for output_dir in "$OUTPUT_ROOT"/*; do
    [[ -d $output_dir/annotation_images ]] || continue
    METHODS+=("${output_dir##*/}")
  done
  ((${#METHODS[@]})) || { echo "No reproduced methods found under: $OUTPUT_ROOT" >&2; exit 1; }

  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  METHODS_FILE="$RUN_LOG_DIR/methods.txt"
  printf '%s\n' "${METHODS[@]}" > "$METHODS_FILE"

  printf 'Submitting metrics for: %s\n' "${METHODS[*]}"
  exec sbatch \
    --job-name=direct_inversion_metrics \
    --account=plgdiffusion3-gpu-a100 \
    --partition=plgrid-gpu-a100 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=128G \
    --gres=gpu:1 \
    --time=13:00:00 \
    --array="0-$((${#METHODS[@]} - 1))%${ARRAY_PARALLELISM}" \
    --output="$RUN_LOG_DIR/%A_%a.out" \
    --error="$RUN_LOG_DIR/%A_%a.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",DATA_PATH="$DATA_PATH",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_LOG_DIR="$RUN_LOG_DIR",METHODS_FILE="$METHODS_FILE" \
    "$0" --worker
fi

[[ -f ${METHODS_FILE:?METHODS_FILE is required in worker mode} ]] || { echo "Missing methods manifest: $METHODS_FILE" >&2; exit 1; }
mapfile -t METHODS < "$METHODS_FILE"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
  echo "Invalid array task id: $TASK_ID" >&2
  exit 2
fi

METHOD=${METHODS[$TASK_ID]}
GENERATED_IMAGES_DIR="$OUTPUT_ROOT/$METHOD/annotation_images"
[[ -d $GENERATED_IMAGES_DIR ]] || { echo "Missing generated images: $GENERATED_IMAGES_DIR" >&2; exit 1; }

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export TORCH_HOME=${TORCH_HOME:-/net/tscratch/people/plgatarsander/torch-cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"

echo "Array task $TASK_ID: $METHOD"
cd "$REPO_DIR"
exec "$PYTHON" -m diff_inversion.eval.pnp_lora_metrics \
  generated_root="$OUTPUT_ROOT" \
  method_name="$METHOD" \
  'result_path=${generated_root}/${method_name}/metrics.csv' \
  'summary_path=${generated_root}/${method_name}/metrics_summary.json' \
  "hydra.run.dir=$RUN_LOG_DIR/hydra/$METHOD" \
  hydra.job.name="direct_inversion_metrics_${METHOD}"
