#!/bin/bash

if [[ -x /net/people/plgrid/plgatarsander/.local/bin/uv ]]; then
  exec /net/people/plgrid/plgatarsander/.local/bin/uv run python diff_inversion/data/generate_sdxl_samples.py \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=sample_gather_train \
    output_dir=/net/tscratch/people/plgatarsander/ZZSN_data/processed/sdxl_trajectories_stacked/train \
    job_id=0,1,2,3,4,5,6,7
fi

echo "'uv' was not found at /net/people/plgrid/plgatarsander/.local/bin/uv." >&2
exit 1
