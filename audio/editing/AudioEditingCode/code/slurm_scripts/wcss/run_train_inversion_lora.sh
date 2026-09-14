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
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_train.sh --array=28    # pair-branch CFG w=2.5
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

# SCRIPT selects the model: the AudioLDM2 trainer by default, or
#   SCRIPT=src/inversion_lora/train_stable_audio.py
# for Stable Audio Open, which carries its own default config. RUN_PREFIX namespaces the
# run so a Stable Audio run cannot write into an AudioLDM2 checkpoint directory of the
# same name, since the two share LORAINV_CHECKPOINT_ROOT and the CONFIGS table below.
SCRIPT="${SCRIPT:-src/inversion_lora/train.py}"
RUN_PREFIX="${RUN_PREFIX:-}"

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
  # saves every 1000 with reconstruction on the same grid, so every checkpoint is picked on
  # reconstruction rather than on loss. The real-audio half of that reconstruction runs at each
  # track's own length capped at 60 s -- the editing pipeline's geometry, 5.9x the 10.24 s the
  # adapter trains on -- over all 35 distinct MedleyDB mixes, since a natural window leaves no
  # offset to vary and more draws would be duplicates. Six evals at ~14 min is ~1.4 h of the
  # 24 h limit: the real set costs less than the old 256 fixed crops did despite the longer
  # windows, because 35 of them replace 256.
  "full|32|16|5e-4|q4_fullte_r32_a16_lr5e-4|train_max_timestep=250 num_loss_bands=5 max_train_steps=6000 save_every_steps=1000 recon_every_steps=1000 recon_real_max_duration_s=60.0 recon_num_real=35"
  # 28: the pair-branch CFG loss. Everything is held at index 26's values -- q4, full preset,
  # r32/a16, lr 5e-4 -- so the only change against a run already scored on the benchmark is the
  # guidance: trajectories generated at w=2.5 with both branches cached, loss and reconstruction
  # both formed at w=2.5. Index 26 fitted a gap 2.98x smaller than the one it was deployed on.
  # Needs the cfg25 dataset: CONFIG_NAME=generate_trajectories_cfg25 through submit_trajectories.sh.
  # Costs 2x forwards per step, so expect roughly double index 26's wall-clock per 1000 steps.
  # batch 16 x accum 2 keeps index 26's effective batch of 32 while holding peak activation memory
  # at one batch-32 forward's worth: both CFG branches stay in the graph, because a merged adapter
  # perturbs both at deployment and the loss on the combination must backprop through each. Batch
  # 32 with two forwards OOMed a 24 GB A5000 at batch 8, and H100 headroom here is untested.
  "full|32|16|5e-4|cfg25_q4_full_r32_a16_lr5e-4|train_max_timestep=250 num_loss_bands=5 max_train_steps=6000 save_every_steps=1000 recon_every_steps=1000 recon_real_max_duration_s=60.0 recon_num_real=35 guidance_scale=2.5 data_root=\${oc.env:LORAINV_DATA_ROOT}/audioldm2_trajectories_cfg25_fp32 batch_size=16 gradient_accumulation_steps=2"
  # 29-34: STABLE AUDIO ONLY. Run with
  #   SCRIPT=src/inversion_lora/train_stable_audio.py RUN_PREFIX=saocos_
  # The AudioLDM2 config has no `cycle` key, so pointing these at train.py aborts on the override.
  #
  # Every field matches the pure-inversion baseline saocos_r8_a4_lr5e-5 (attn, r8/a4, 5e-5), so
  # the cycle term is the only difference, and these runs' step-4000 checkpoints line up with the
  # baseline checkpoint the hparam sweeps already scored.
  #
  # 29-32: k=1 at four gradient-balance targets. lambda is set per step so the cycle term's
  # gradient norm is target_ratio x the inversion term's. On this grid B^2 is constant at 5.21e-3
  # (geometric sigmas => step-independent A, B), so the term contributes no schedule reweighting
  # at all: its only content is the bootstrapped target plus the gradient path through the frozen
  # teacher. 12000 steps rather than 20000 because the extra teacher forward and its backward put
  # the step near 2x the baseline's, and 20000 would run past the 24 h limit.
  "attn|8|4|5e-5|cyc_k1_g0.1_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=1 cycle.target_ratio=0.1 max_train_steps=12000"
  "attn|8|4|5e-5|cyc_k1_g0.5_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=1 cycle.target_ratio=0.5 max_train_steps=12000"
  "attn|8|4|5e-5|cyc_k1_g1.0_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=1 cycle.target_ratio=1.0 max_train_steps=12000"
  "attn|8|4|5e-5|cyc_k1_g2.0_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=1 cycle.target_ratio=2.0 max_train_steps=12000"
  # 33-34: the multi-step cycle, which is the only variant that penalises how per-step residuals
  # COMPOUND -- a one-step round trip cannot see that. k inversion steps (only the last carrying
  # gradient) then k generation steps with the frozen teacher, so k + 1 graphs are alive at once;
  # gradient checkpointing cannot buy that room back (the recompute would run with the adapter on),
  # so batch 4 x accum 8 holds the effective batch at 32. 6000 steps: at ~3-4x the baseline step
  # this is the most that fits the walltime, and checkpoints land every 2000 either way.
  "attn|8|4|5e-5|cyc_k2_g1.0_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=2 cycle.target_ratio=1.0 max_train_steps=6000 batch_size=4 gradient_accumulation_steps=8"
  "attn|8|4|5e-5|cyc_k3_g1.0_r8_a4_lr5e-5|cycle.enabled=true cycle.steps=3 cycle.target_ratio=1.0 max_train_steps=6000 batch_size=4 gradient_accumulation_steps=8"
  # 35: EXPERIMENT 1, Stable Audio only. The adapter is trained at w=1 but deployed at cfg_tar
  # 3.5-7.0, and the gap it must close grows ~3x from guidance 1.0 to 2.5 -- so it has been
  # fitting a gap several times smaller than the one it meets. Everything matches the baseline
  # saocos_r8_a4_lr5e-5 except the guidance, so the comparison is clean.
  # Needs the cfg35 dataset: CONFIG_NAME=generate_trajectories_stable_audio_cfg35 first.
  # Two forwards per step for the student plus the guided target, so ~2x the baseline step;
  # batch 4 x accum 8 holds the effective batch at 32 while keeping one batch-4 graph per branch.
  # 2000 steps, not 12000: the accuracy ladder showed editing dead flat from step 2000 to 20000,
  # and at ~10 s/step 12000 would be killed on walltime at ~8600 anyway -- which would also make
  # a chained edit sweep wait 24 h for a checkpoint that exists at 5.6 h. Saving every 500 gives
  # a four-point dose-response inside the run for free.
  "attn|8|4|5e-5|cfg35_r8_a4_lr5e-5|guidance_scale=3.5 data_root=\${oc.env:LORAINV_DATA_ROOT}/stable_audio_cosine_ode_cfg35_fp32 max_train_steps=2000 save_every_steps=500 eval_every_steps=500 batch_size=4 gradient_accumulation_steps=8"
  # 36: EXPERIMENT H2, Stable Audio only. Same objective and grid as the baseline
  # saocos_r8_a4_lr5e-5; the only difference is the dataset: trajectories sampled with 991 fine
  # steps and reduced to the 100-point coarse grid with exact-coarse-step inputs
  # (generate_trajectories_stable_audio_dense.yaml), so the inputs sit on a higher-quality
  # trajectory. Needs CONFIG_NAME=generate_trajectories_stable_audio_dense first. 3000 steps:
  # editing is flat from step 2000, and saving every 500 gives the dose-response for free.
  "attn|8|4|5e-5|dense991_r8_a4_lr5e-5|data_root=\${oc.env:LORAINV_DATA_ROOT}/stable_audio_cosine_ode_fp32_dense991 max_train_steps=3000 save_every_steps=500 eval_every_steps=500"
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
RUN_NAME="$RUN_PREFIX$RUN_NAME"
echo "task=$TASK_ID preset=$PRESET rank=$RANK alpha=$ALPHA lr=$LR run=$RUN_NAME node=$(hostname)"
echo "extra overrides: ${EXTRA:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$SCRIPT" \
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
