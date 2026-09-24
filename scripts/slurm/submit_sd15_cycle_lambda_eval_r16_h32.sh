#!/bin/bash

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_DIR}"
mkdir -p logs/slurm

for TASK_ID in 1 3 5 7; do
  for LAMBDA in 0.5 2 4 8 16; do
  TAG=${LAMBDA/./p}
  CHECKPOINT=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_lambda${TAG}_bs120_e1/checkpoint_final.pt
  RUN_TAG=cycle_loss_cfg1_r16_lambda${TAG}_uniform_e1_all_editors
  [[ -s "${CHECKPOINT}" ]] || { echo "Missing final checkpoint: ${CHECKPOINT}" >&2; exit 2; }

    JOB_ID=$(sbatch --parsable \
      --job-name="cyc-l${TAG}-t${TASK_ID}" \
      scripts/slurm/eval_all_editors_lora_full_h32.sbatch \
      "${CHECKPOINT}" 16 "${RUN_TAG}" "${TASK_ID}" 1.0 single)
    echo "lambda=${LAMBDA} task=${TASK_ID} job=${JOB_ID}"
  done
done
