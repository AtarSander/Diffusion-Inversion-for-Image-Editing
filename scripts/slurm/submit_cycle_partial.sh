#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/slurm
# Independent ordinary jobs: no arrays or dependencies between editors/depths.
# Restrict either partition to h32/h86; glasser is excluded explicitly.
for editor in p2p pnp masactrl pix2pix_zero; do
  for percent in 94 96 98; do
    sbatch --parsable --exclude=glasser \
      --job-name="cyc-r8-${editor}-p${percent}" \
      scripts/slurm/eval_cycle_partial.sbatch "$percent" "$editor"
  done
done
