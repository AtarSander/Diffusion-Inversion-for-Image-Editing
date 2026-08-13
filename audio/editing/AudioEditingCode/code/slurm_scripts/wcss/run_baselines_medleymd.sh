#!/bin/bash -l
# ABOUTME: PWR/WCSS array job reproducing the six MedleyMD editing baselines (DDPM-inv,
# ABOUTME: DDIM-inv, SDEdit) for AudioLDM2 and Stable Audio over the full 696-row benchmark.
#SBATCH --job-name=medleymd-baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-83
#SBATCH --output=outputs/logs/slurm/baselines-%A_%a.out
#SBATCH --error=outputs/logs/slurm/baselines-%A_%a.err
#
# Submit from the audio/ directory:
#   cd /lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/code/lorainv/audio
#   mkdir -p outputs/logs/slurm
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_baselines_medleymd.sh
#
# Prerequisites, once, on the login node:
#   module load Python/3.10.4-GCCcore-11.3.0
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements_lorainv.txt
#   python editing/AudioEditingCode/code/slurm_scripts/wcss/prefetch_models.py
#   # and a .env holding MEDLEYDB_AUDIO_DIR, HF_TOKEN, HF_HOME, EDIT_OUTPUTS_DIR

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

# lem compute nodes have internet, so a cache miss downloads rather than failing. Still run
# prefetch_models.py first: 72 tasks each pulling the same ~12 GB is slow and rate-limit prone.
# Set HF_HUB_OFFLINE=1 when submitting if you want a cold cache to fail fast instead.

CODE_DIR="editing/AudioEditingCode/code"
N_PARTS=12

# script|mode|steps|cfg_src|cfg_tar|tstart|run_name
# DDIM uses tstart == steps: partial inversion is not DDIM inversion, and this is the exact
# baseline the LoRA inversion is meant to improve.
CONFIGS=(
  "edit_audioldm_medleydb.py|ddpm|200|3.0|12.0|100|audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200"
  "edit_audioldm_medleydb.py|ddim|200|3.0|12.0|200|audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200"
  "edit_audioldm_medleydb.py|sdedit|200|3.0|12.0|100|audioldm2_sdedit_cfgtar12.0_t100_s200"
  "edit_stableaudio_medleydb.py|ddpm|100|1.0|3.5|50|stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100"
  "edit_stableaudio_medleydb.py|ddim|100|1.0|3.5|100|stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100"
  "edit_stableaudio_medleydb.py|sdedit|100|1.0|3.5|50|stableaudio_sdedit_cfgtar3.5_t50_s100"
  # Index 6 (array 72-83): DDIM inversion with no CFG on the forward pass. DDIM inversion is
  # known to degrade as the inversion-side guidance grows, so this separates "DDIM inversion is
  # weak" from "DDIM inversion is weak at cfg_src=3.0" -- which decides whether the inversion
  # LoRA should target discretisation error or the CFG mismatch.
  "edit_audioldm_medleydb.py|ddim|200|1.0|12.0|200|audioldm2_ddim_cfgsrc1.0_cfgtar12.0_t200_s200"
)

# Index 7 (array 84-95): DDIM inversion with a trained inversion LoRA on the forward pass only.
# Kept out of CONFIGS because it needs an extra argument; set LORA_PATH to a checkpoint written
# by src/inversion_lora/train.py. cfg_src=1.0 matches the No-CFG objective the adapter was
# trained under, and the cfg_src=1.0 row above is its like-for-like comparison.
LORA_CONFIG="edit_audioldm_medleydb.py|ddim|200|1.0|12.0|200"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
CFG_IDX=$(( TASK_ID / N_PARTS ))
PART=$(( TASK_ID % N_PARTS ))
if [ "$CFG_IDX" -gt "${#CONFIGS[@]}" ]; then
  echo "array index $TASK_ID exceeds ${#CONFIGS[@]} configs (+1 LoRA) x $N_PARTS shards" >&2
  exit 2
fi

EXTRA_ARGS=()
if [ "$CFG_IDX" -eq "${#CONFIGS[@]}" ]; then
  : "${LORA_PATH:?set LORA_PATH=<checkpoint.pt> to run the LoRA config (array 84-95)}"
  IFS='|' read -r SCRIPT MODE STEPS CFG_SRC CFG_TAR TSTART <<< "$LORA_CONFIG"
  source "$(dirname "${BASH_SOURCE[0]}")/lora_run_name.sh"
  RUN_NAME="$(lora_run_name "$LORA_PATH")"
  EXTRA_ARGS=(--lora_path "$LORA_PATH")
else
  IFS='|' read -r SCRIPT MODE STEPS CFG_SRC CFG_TAR TSTART RUN_NAME <<< "${CONFIGS[$CFG_IDX]}"
fi
echo "task=$TASK_ID config=$RUN_NAME shard=$PART/$N_PARTS node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$CODE_DIR"

# Transient filesystem/network failures are common here, so retry like the other WCSS jobs.
ok=0
for attempt in 1 2 3; do
  python "$SCRIPT" \
    --mode "$MODE" \
    --num_diffusion_steps "$STEPS" \
    --cfg_src "$CFG_SRC" \
    --cfg_tar "$CFG_TAR" \
    --tstart "$TSTART" \
    --seed 42 \
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
