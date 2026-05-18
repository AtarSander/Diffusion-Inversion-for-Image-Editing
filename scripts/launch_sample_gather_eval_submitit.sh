#!/bin/bash

if [[ -x /net/people/plgrid/plgatarsander/.local/bin/uv ]]; then
  exec /net/people/plgrid/plgatarsander/.local/bin/uv run python diff_inversion/data/generate_sdxl_samples.py \
    --config-name sample_gather_eval_submitit \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=sample_gather_eval \
    job_id=0,1
fi

echo "'uv' was not found at /net/people/plgrid/plgatarsander/.local/bin/uv." >&2
exit 1
