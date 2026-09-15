# ABOUTME: The w=1 adapter's missing cfg_tar=7.0 cells at step 2000, so it can be compared against
# ABOUTME: the CFG adapter at the same target guidance rather than only at cfg_tar 3.5.

# step_4000 already has cfg_tar 7.0 on disk; step_2000 does not, and 2000 is the checkpoint the
# CFG adapter and the cycle arms were all scored at, so this completes the matched comparison.
LORA_MODE=odeinv
LORA_CHECKPOINTS=("saocos_r8_a4_lr5e-5/checkpoint_step_2000.pt")
LORA_TSTART=(25 50 75 99)
LORA_CFG_TAR=(7.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
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

# Deliberately the sao_lora_sweep_configs.sh naming, with NO cfgsrc field: these are cfg_src=1.0
# like every other arm on the main curves, and adding a cfgsrc field would split one arm into two
# in the plotter rather than extending the existing one.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  echo "stableaudio_odeinvlora_${LORA_SPLIT}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
