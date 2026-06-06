#!/bin/bash

PYTHON_BIN=/net/tscratch/people/plgmichalsadowski/venv/bin/python3

if [[ -x "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" diff_inversion/data/precompute_training_cache.py \
    --config-name precompute_training_cache_submitit \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=precompute_training_cache_train \
    job_id=0,1,2,3,4,5,6,7
fi

echo "Python was not found at $PYTHON_BIN." >&2
exit 1
