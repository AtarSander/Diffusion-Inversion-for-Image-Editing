#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/people/plgrid/plgatarsander/Diffusion-Inversion-for-Image-Editing}

for arg in "$@"; do
  if [[ "$arg" == "hydra/launcher=basic" && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing hydra/launcher=basic outside a Slurm allocation." >&2
    echo "Use this launcher without hydra/launcher=basic, or run scripts/submit_sdxl_eval_chain.sh <r8|r16|r32>." >&2
    exit 2
  fi
done

if [[ "${REQUIRE_EVAL_ARTIFACT_DIR:-true}" == "true" ]]; then
  has_artifact_dir=false
  for arg in "$@"; do
    if [[ "$arg" == artifact_dir=* || "$arg" == +artifact_dir=* ]]; then
      has_artifact_dir=true
      break
    fi
  done
  if [[ "$has_artifact_dir" != "true" ]]; then
    echo "Refusing in-place eval: pass artifact_dir=... or use scripts/submit_sdxl_eval_chain.sh <r8|r16|r32>." >&2
    echo "Set REQUIRE_EVAL_ARTIFACT_DIR=false only if you intentionally want to read/write input_dir/sample_* artifacts." >&2
    exit 2
  fi
fi

source "$REPO_DIR/scripts/eval_env.sh"
setup_eval_environment
preflight_eval_environment

exec "${EVAL_PYTHON_CMD[@]}" diff_inversion/eval/run.py \
  --config-name eval_run_submitit \
  --multirun \
  'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.name=sdxl_eval_run \
  "$@"
