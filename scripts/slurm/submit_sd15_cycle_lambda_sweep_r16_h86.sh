#!/bin/bash

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/diagnostics

DIAGNOSTIC_JOB=$(sbatch --parsable scripts/slurm/diagnose_sd15_cycle_gradients_h86.sbatch)
echo "gradient diagnostic: ${DIAGNOSTIC_JOB}"

for LAMBDA in 0.5 2 4 8 16; do
  TAG=${LAMBDA/./p}
  RUN_NAME=sd15-inversion-lora-r16-conditional-cycle-cfg1-lambda${TAG}-bs120-e1
  CHECKPOINT_DIR=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_lambda${TAG}_bs120_e1
  COMMON_OVERRIDES=(
    training_target=conditional_cycle
    "training_target.recon_lambda=${LAMBDA}"
    sampling=uniform
    "run_name=${RUN_NAME}"
    "checkpoint_dir=${CHECKPOINT_DIR}"
    data.batch_size=120
    max_train_steps=12684
    lr_scheduler.warmup_steps=254
    save_every_steps=500
    eval_every_steps=0
    max_val_batches=0
    validation_preview.enabled=false
  )

  FIRST_JOB=$(sbatch --parsable \
    --job-name="cyc-r16-l${TAG}-1" \
    scripts/slurm/train_sd15_lora_h86.sbatch 16 "${COMMON_OVERRIDES[@]}")
  SECOND_JOB=$(sbatch --parsable \
    --dependency="afterany:${FIRST_JOB}" \
    --job-name="cyc-r16-l${TAG}-2" \
    --export=ALL,AUTO_RESUME=1 \
    scripts/slurm/train_sd15_lora_h86.sbatch 16 "${COMMON_OVERRIDES[@]}")
  THIRD_JOB=$(sbatch --parsable \
    --dependency="afterany:${SECOND_JOB}" \
    --job-name="cyc-r16-l${TAG}-3" \
    --export=ALL,AUTO_RESUME=1 \
    scripts/slurm/train_sd15_lora_h86.sbatch 16 "${COMMON_OVERRIDES[@]}")

  echo "lambda ${LAMBDA}: ${FIRST_JOB} -> ${SECOND_JOB} -> ${THIRD_JOB}"
done
