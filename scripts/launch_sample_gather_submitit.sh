#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/net/pr2/projects/plgrid/plggzzsn2026/diffusion_inversion/Diffusion-Inversion-for-Image-Editing}
PYTHON=${PYTHON:-/net/tscratch/people/plgmichalsadowski/conda_envs/diff-inversion/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-/net/tscratch/people/plgmichalsadowski/ZZSN_data/processed/sdxl_trajectories_stacked/train}
JOB_IDS=${JOB_IDS:-0,1,2,3,4,5,6,7}

cd "$REPO_DIR"

module load Miniconda3/23.3.1-0

export HF_HOME=${HF_HOME:-/net/tscratch/people/plgmichalsadowski/hf-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/net/tscratch/people/plgmichalsadowski/wandb-cache}
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"

exec "$PYTHON" diff_inversion/data/generate_sdxl_samples.py \
  --config-name sample_gather_submitit \
  --multirun \
  'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.name=sample_gather_train \
  output_dir="$OUTPUT_DIR" \
  "job_id=$JOB_IDS"
