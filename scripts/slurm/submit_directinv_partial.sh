#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/slurm
for editor in p2p pnp masactrl pix2pix_zero; do
  for percent in 90 92 94 96 98; do
    sbatch --parsable --exclude=glasser --job-name="di-${editor}-p${percent}" \
      scripts/slurm/eval_directinv_partial.sbatch "$percent" "$editor"
  done
done
