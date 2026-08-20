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
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=0-5   # rank+lr cross
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=6-11  # lr sweep
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=12-23 # rerun + conv/ff
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=24-26 # t<=250 only
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=27    # full+time_emb
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
# preset|rank|alpha|learning_rate|run_name
# preset selects which module families the adapter touches (see lora_target_presets in the
# config): attn is attention projections only, attn_ff adds the feed-forward layers, full adds
# the transformer in/out projections and the ResNet convs.
CONFIGS=(
  # 0-5: the original rank + learning-rate cross, scored on the old 32-sample eval.
  "attn|8|4|5e-5|r8_a4_lr5e-5"
  "attn|8|4|2e-4|r8_a4_lr2e-4"
  "attn|8|4|1e-5|r8_a4_lr1e-5"
  "attn|4|2|2e-4|r4_a2_lr2e-4"
  "attn|16|8|2e-4|r16_a8_lr2e-4"
  "attn|32|16|2e-4|r32_a16_lr2e-4"
  # 6-11: learning rates above 2e-4. Found the plateau (3e-4..2e-3 identical) and the ceiling
  # (5e-3 diverges).
  "attn|8|4|3e-4|r8_a4_lr3e-4"
  "attn|8|4|5e-4|r8_a4_lr5e-4"
  "attn|8|4|1e-3|r8_a4_lr1e-3"
  "attn|8|4|2e-3|r8_a4_lr2e-3"
  "attn|8|4|5e-3|r8_a4_lr5e-3"
  "attn|8|4|1e-2|r8_a4_lr1e-2"
  # 12-17: the 0-5 cross re-run, so it is measured by the 256-sample eval the later runs use.
  # Names carry the preset, so these do not collide with the originals in W&B.
  "attn|8|4|5e-5|attn_r8_a4_lr5e-5"
  "attn|8|4|2e-4|attn_r8_a4_lr2e-4"
  "attn|8|4|1e-5|attn_r8_a4_lr1e-5"
  "attn|4|2|2e-4|attn_r4_a2_lr2e-4"
  "attn|16|8|2e-4|attn_r16_a8_lr2e-4"
  "attn|32|16|2e-4|attn_r32_a16_lr2e-4"
  # 18-23: more of the network, at the learning rate the sweep settled on. Rank is varied within
  # each preset because adding module families may change where rank starts to bind, which it
  # did not for attention alone.
  "attn_ff|8|4|5e-4|attnff_r8_a4_lr5e-4"
  "attn_ff|16|8|5e-4|attnff_r16_a8_lr5e-4"
  "attn_ff|32|16|5e-4|attnff_r32_a16_lr5e-4"
  "full|8|4|5e-4|full_r8_a4_lr5e-4"
  "full|16|8|5e-4|full_r16_a8_lr5e-4"
  "full|32|16|5e-4|full_r32_a16_lr5e-4"
  # 24-26: train only on the cleanest quarter of the schedule (t <= 250), where the shift gap is
  # ~300x what it is at the noisy end, with the loss split into 5% bands (q25_20..q05_00). Same
  # 20k steps, but each one now lands where the error is; an epoch is a quarter as long, so this
  # is ~9 epochs over 71k transitions rather than 2.2 over 285k.
  "attn|8|4|5e-4|q4_attn_r8_a4_lr5e-4|train_max_timestep=250 num_loss_bands=5"
  "attn|32|16|5e-4|q4_attn_r32_a16_lr5e-4|train_max_timestep=250 num_loss_bands=5"
  "full|32|16|5e-4|q4_full_r32_a16_lr5e-4|train_max_timestep=250 num_loss_bands=5"
  # 27: index 26 re-run now that the full preset also adapts the timestep-embedding modules
  # (time_embedding.linear_1/_2 and the 22 per-ResNet time_emb_proj, 1467 -> 1491 modules). The
  # shift gap the adapter has to close is strongly t-dependent, and until now nothing the adapter
  # touched saw the timestep directly. Stops at 6000 steps -- q4 reconstruction peaked near 7500
  # and the scored index-26 checkpoint was step 6000, so this is the matched comparison -- and
  # saves every 1000 so a checkpoint can be picked on reconstruction rather than on loss.
  "full|32|16|5e-4|q4_fullte_r32_a16_lr5e-4|train_max_timestep=250 num_loss_bands=5 max_train_steps=6000 save_every_steps=1000"
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

# EXTRA is optional: space-separated Hydra overrides for configs that need more than the four
# fields above, so the entries that do not need any stay untouched.
IFS='|' read -r PRESET RANK ALPHA LR RUN_NAME EXTRA <<< "${CONFIGS[$TASK_ID]}"
echo "task=$TASK_ID preset=$PRESET rank=$RANK alpha=$ALPHA lr=$LR run=$RUN_NAME node=$(hostname)"
echo "extra overrides: ${EXTRA:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/inversion_lora/train.py \
  device=cuda:0 \
  lora_preset="$PRESET" \
  lora.r="$RANK" \
  lora.lora_alpha="$ALPHA" \
  learning_rate="$LR" \
  run_name="$RUN_NAME" \
  wandb_mode=online \
  checkpoint_dir="${LORAINV_CHECKPOINT_ROOT:?not set or empty in .env}/$RUN_NAME" \
  ${EXTRA:-}
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: $RUN_NAME" >&2
  exit "$rc"
fi
echo "done: $RUN_NAME"
