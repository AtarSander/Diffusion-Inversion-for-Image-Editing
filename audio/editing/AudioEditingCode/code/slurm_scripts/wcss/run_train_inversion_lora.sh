#!/bin/bash -l
# ABOUTME: PWR/WCSS array job training the AudioLDM2 inversion LoRA, one array task per
# ABOUTME: hyperparameter combination, each logging to W&B online under its own run name.
#SBATCH --job-name=lorainv-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-5
#SBATCH --output=outputs/logs/slurm/train-%A_%a.out
#SBATCH --error=outputs/logs/slurm/train-%A_%a.err
#
# Submit from audio/ once the trajectory array has finished and been verified:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh
#
# Prerequisites:
#   WANDB_API_KEY in ~/.bashrc (alongside HF_TOKEN; it is a secret, so not in .env). `wandb login`
#   needs the venv, which is not available on the login node, so the key is the way in.
#   The trajectory dataset must have passed run_verify_trajectories.sh.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# rank|alpha|learning_rate|run_name
# Rank and learning rate are the two that decide whether the adapter has the capacity to close
# the shift gap at all; alpha tracks rank so the effective scale alpha/r stays comparable.
CONFIGS=(
  "8|4|5e-5|r8_a4_lr5e-5"
  "8|4|2e-4|r8_a4_lr2e-4"
  "8|4|1e-5|r8_a4_lr1e-5"
  "4|2|2e-4|r4_a2_lr2e-4"
  "16|8|2e-4|r16_a8_lr2e-4"
  "32|16|2e-4|r32_a16_lr2e-4"
)

# Fail before the 12 GB model load rather than after it: wandb only reports a bad credential
# once it tries to sync, by which point the job has burned several minutes.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "ERROR: no W&B credential. Add WANDB_API_KEY to ~/.bashrc (next to HF_TOKEN), or pass" >&2
  echo "  wandb_mode=offline to this script and sync the runs later." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
if [ "$TASK_ID" -ge "${#CONFIGS[@]}" ]; then
  echo "array index $TASK_ID exceeds ${#CONFIGS[@]} configs" >&2
  exit 2
fi

IFS='|' read -r RANK ALPHA LR RUN_NAME <<< "${CONFIGS[$TASK_ID]}"
echo "task=$TASK_ID rank=$RANK alpha=$ALPHA lr=$LR run=$RUN_NAME node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/inversion_lora/train.py \
  device=cuda:0 \
  lora.r="$RANK" \
  lora.lora_alpha="$ALPHA" \
  learning_rate="$LR" \
  run_name="$RUN_NAME" \
  wandb_mode=online \
  checkpoint_dir="${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}/$RUN_NAME"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: $RUN_NAME" >&2
  exit "$rc"
fi
echo "done: $RUN_NAME"
