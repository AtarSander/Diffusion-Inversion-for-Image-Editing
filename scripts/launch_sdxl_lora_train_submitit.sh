#!/bin/bash

if [[ -x /net/people/plgrid/plgatarsander/.local/bin/uv ]]; then
  exec /net/people/plgrid/plgatarsander/.local/bin/uv run python diff_inversion/modeling/train.py \
    --config-name train_submitit \
    --multirun \
    'hydra.sweep.dir=slurm_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.name=sdxl_lora_train
fi

echo "'uv' was not found at /net/people/plgrid/plgatarsander/.local/bin/uv." >&2
exit 1
