#!/bin/bash -l
# ABOUTME: PWR/WCSS array job reconstructing the 35 distinct MedleyDB tracks through DDIM
# ABOUTME: inversion at two step counts, with and without a trained inversion LoRA.
#SBATCH --job-name=medleydb-recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-35
#SBATCH --output=outputs/logs/slurm/recon-%A_%a.out
#SBATCH --error=outputs/logs/slurm/recon-%A_%a.err
#
# Submit from audio/:
#   export LORA_ATTN=$LORAINV_CHECKPOINT_ROOT/attn_r8_a4_lr2e-4/checkpoint_step_6000.pt
#   export LORA_FULL=$LORAINV_CHECKPOINT_ROOT/full_r32_a16_lr5e-4/checkpoint_step_6000.pt
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_reconstruct.sh
#
# Reconstruction is the editing pipeline with the target caption replaced by the source caption
# and cfg_tar=1.0, so the output should be the input. That isolates how exactly DDIM inversion
# round-trips real audio, which is what the LoRA is trained to improve -- unlike the edit
# benchmark, where the reverse pass runs a different prompt at cfg_tar=12.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

CODE_DIR="editing/AudioEditingCode/code"
N_PARTS=6

# steps|lora_env_var|run_suffix   (empty lora_env_var means the frozen teacher)
CONFIGS=(
  "200||nolora"
  "200|LORA_ATTN|attn"
  "200|LORA_FULL|full"
  "50||nolora"
  "50|LORA_ATTN|attn"
  "50|LORA_FULL|full"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
CFG_IDX=$(( TASK_ID / N_PARTS ))
PART=$(( TASK_ID % N_PARTS ))
if [ "$CFG_IDX" -ge "${#CONFIGS[@]}" ]; then
  echo "array index $TASK_ID exceeds ${#CONFIGS[@]} configs x $N_PARTS shards" >&2
  exit 2
fi

IFS='|' read -r STEPS LORA_VAR SUFFIX <<< "${CONFIGS[$CFG_IDX]}"

EXTRA_ARGS=()
if [ -n "$LORA_VAR" ]; then
  LORA_PATH="${!LORA_VAR:?$LORA_VAR is not set; export it before submitting}"
  EXTRA_ARGS=(--lora_path "$LORA_PATH")
fi
RUN_NAME="recon_tracks_s${STEPS}_${SUFFIX}"
: "${RUN_NAME:?run name resolved empty}"

echo "task=$TASK_ID steps=$STEPS lora=${LORA_VAR:-none} run=$RUN_NAME shard=$PART/$N_PARTS node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$CODE_DIR"

ok=0
for attempt in 1 2 3; do
  python edit_audioldm_medleydb.py \
    --mode ddim \
    --num_diffusion_steps "$STEPS" \
    --cfg_src 1.0 \
    --cfg_tar 1.0 \
    --tstart "$STEPS" \
    --seed 42 \
    --unique_tracks True \
    --reconstruct True \
    --n_parts "$N_PARTS" \
    --part_id "$PART" \
    --run_name "$RUN_NAME" "${EXTRA_ARGS[@]}" && { ok=1; break; }
  echo "attempt $attempt failed for $RUN_NAME shard $PART; retrying in 120s..." >&2
  sleep 120
done

if [ "$ok" != 1 ]; then
  echo "FAILED 3x: $RUN_NAME shard $PART" >&2
  exit 1
fi
echo "done: $RUN_NAME shard $PART"
