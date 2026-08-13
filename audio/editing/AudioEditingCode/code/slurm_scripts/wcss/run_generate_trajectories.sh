#!/bin/bash -l
# ABOUTME: PWR/WCSS array job generating the AudioLDM2 DDIM trajectory dataset for the inversion
# ABOUTME: LoRA. Each task owns a contiguous slice of the caption list and writes its own samples.
#SBATCH --job-name=lorainv-trajectories
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-7
#SBATCH --output=outputs/logs/slurm/traj-%A_%a.out
#SBATCH --error=outputs/logs/slurm/traj-%A_%a.err
#
# Submit from audio/:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_trajectories.sh
#
# Samples are independent directories, and meta.json is written last as the completion
# sentinel, so tasks never collide and a re-run skips whatever already finished.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1500}"
N_SHARDS="${N_SHARDS:-8}"
CONFIG_NAME="${CONFIG_NAME:-generate_trajectories_gonogo}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
PER_SHARD=$(( (TOTAL_SAMPLES + N_SHARDS - 1) / N_SHARDS ))
START=$(( TASK_ID * PER_SHARD ))
# The last shard is short whenever TOTAL_SAMPLES is not a multiple of N_SHARDS.
COUNT=$(( TOTAL_SAMPLES - START ))
if [ "$COUNT" -gt "$PER_SHARD" ]; then COUNT=$PER_SHARD; fi
if [ "$COUNT" -le 0 ]; then
  echo "shard $TASK_ID has no work (total=$TOTAL_SAMPLES shards=$N_SHARDS)"
  exit 0
fi

echo "task=$TASK_ID start_index=$START num_samples=$COUNT node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/inversion_lora/generate_trajectories.py \
  --config-name "$CONFIG_NAME" \
  device=cuda:0 \
  start_index="$START" \
  num_samples="$COUNT"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: shard $TASK_ID" >&2
  exit "$rc"
fi
echo "done: shard $TASK_ID (start=$START count=$COUNT)"
