# ABOUTME: Experiment 1 -- the CFG-aware adapter on the editing benchmark, swept at both inversion
# ABOUTME: guidances so "trained at w=3.5" is separable from "inverted at w=3.5".

# The adapter is trained on guided predictions (w=3.5), but the editing pipeline inverts at
# cfg_src=1.0 by default. Deploying it unguided recreates the train/deploy mismatch in the other
# direction and measures nothing, so both are run:
#   SRC=3.5  matched to training -- the actual test. Inversion runs both branches, so its NFE is
#            roughly double every other arm's; say so before putting it on a matched-NFE plot.
#   SRC=1.0  identical to every existing arm, so the comparison to the current curve is clean.
SRC="${SRC:?set SRC=1.0 or 3.5}"

LORA_MODE=odeinv
# Step 2000: the accuracy ladder showed editing flat from 2000 to 20000, and at ~10 s/step this
# run cannot reach 12000 inside the walltime anyway.
LORA_CHECKPOINTS=(
  "saocos_cfg35_r8_a4_lr5e-5/checkpoint_step_2000.pt"
)
LORA_TSTART=(25 50 75 99)
LORA_CFG_TAR=(3.5)
LORA_STEPS=100
LORA_CFG_SRC="$SRC"
LORA_SPLIT=hparam

lora_sweep_configs() {
  local ckpt tstart cfg
  for ckpt in "${LORA_CHECKPOINTS[@]}"; do
    for tstart in "${LORA_TSTART[@]}"; do
      for cfg in "${LORA_CFG_TAR[@]}"; do
        echo "$ckpt|$tstart|$cfg"
      done
    done
  done
}

# cfg_src is in the name: the two arms differ only by it, so without it they collide and one
# silently overwrites the other.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  echo "stableaudio_odeinvlora_${LORA_SPLIT}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_cfgsrc${LORA_CFG_SRC}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
