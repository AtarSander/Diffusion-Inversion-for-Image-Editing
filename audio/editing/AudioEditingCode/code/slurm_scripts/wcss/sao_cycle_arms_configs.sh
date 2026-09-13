# ABOUTME: Does the MULTI-step cycle loss change editing? The k=2 and k=3 arms against the plain
# ABOUTME: inversion baseline, all at step 2000, which is the deepest checkpoint every arm reached.

# The k=1 arms are deliberately absent. Four of them, spanning 20x in gradient-balance target,
# matched the no-cycle baseline to 6-7 significant figures at every checkpoint (relative spread
# 3.4e-6 at step 4000; see output/cycle_arms/). grad L_cycle is parallel to grad L_inv up to a
# (B/A)||J|| rotation with B/A = 0.0778, and Adam is per-parameter scale invariant, so a parallel
# gradient changes no update. Editing them would re-measure saocos_r8_a4_lr5e-5 at cost.
LORA_MODE=odeinv

# Step 2000 for all three: k=3 runs at ~26.8 s/step and cannot reach 4000 inside the walltime, so
# 2000 is the only depth where the comparison is matched. The baseline entry is its OWN step-2000
# checkpoint, not the step-4000 one the earlier sweeps used -- comparing arms at different depths
# would confound the cycle term with training length.
LORA_CHECKPOINTS=(
  "saocos_r8_a4_lr5e-5/checkpoint_step_2000.pt"
  "saocos_cyc_k2_g1.0_r8_a4_lr5e-5/checkpoint_step_2000.pt"
  "saocos_cyc_k3_g1.0_r8_a4_lr5e-5/checkpoint_step_2000.pt"
)

# The no-LoRA arm is not re-run: stableaudio_odeinv_nolora_hparam_cfgtar3.5_t*_s100 is already on
# disk from the earlier sweep at identical settings, and pairs row-for-row with these.
LORA_TSTART=(25 50 75 99)
LORA_CFG_TAR=(3.5)   # one guidance value: this asks whether the arms differ, not where the front is
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

lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  if [ -z "$ckpt" ]; then
    echo "stableaudio_odeinv_nolora_${LORA_SPLIT}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
    return
  fi
  echo "stableaudio_odeinvlora_${LORA_SPLIT}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
