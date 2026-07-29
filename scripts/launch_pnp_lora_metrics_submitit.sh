#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-$REPO_DIR/.venv-pnp-metrics/bin/python}
UV=${UV:-uv}
if [[ -z "${HYDRA_SWEEP_DIR:-}" ]]; then
  HYDRA_SWEEP_DIR='/net/pr2/projects/plgrid/plggdiffusion/plgatarsander/logs/pnp_lora_metrics/${now:%Y-%m-%d}/${now:%H-%M-%S}'
fi

cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

export UV_CACHE_DIR=${UV_CACHE_DIR:-/net/tscratch/people/plgatarsander/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO_DIR/.venv}
export HF_HOME=${HF_HOME:-/net/tscratch/people/plgatarsander/hf-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/net/tscratch/people/plgatarsander/matplotlib-cache}
export TORCH_HOME=${TORCH_HOME:-/net/tscratch/people/plgatarsander/torch-cache}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"

if [[ -x "$PYTHON" ]]; then
  exec "$PYTHON" -m diff_inversion.eval.pnp_lora_metrics \
    --config-name eval_pnp_lora_metrics_submitit \
    --multirun \
    "hydra.sweep.dir=$HYDRA_SWEEP_DIR" \
    hydra.job.name=pnp_lora_metrics \
    "$@"
fi

exec "$UV" run --frozen python -m diff_inversion.eval.pnp_lora_metrics \
  --config-name eval_pnp_lora_metrics_submitit \
  --multirun \
  "hydra.sweep.dir=$HYDRA_SWEEP_DIR" \
  hydra.job.name=pnp_lora_metrics \
  "$@"
