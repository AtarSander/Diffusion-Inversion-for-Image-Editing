# ABOUTME: Re-run Stable Audio's inversion baselines on the corrected sampler: native cosine sigma
# ABOUTME: grid, first-order ODE, exact algebraic inverse, replacing the rejected `ddim` numbers.

# The `ddim` baselines in the benchmark tables were produced by a sampler that rebuilds the
# scheduler onto a linear-beta grid, querying the DiT ~1000x outside its trained timestep range:
# data predictions ~100x too large, decoded audio clipping by 22x, and an inverse scheduler that is
# not the inverse of its own reverse pass. See output/sao_schedules/REPORT.md. This grid replaces
# those numbers. The `ddpm` and `sdedit` baselines use the native scheduler and stand.
LORA_MODE=odeinv
LORA_CHECKPOINTS=("")  # no adapter: these are baselines

# tstart 99 is full inversion -- the last reverse step ends at sigma = 0, discards the sample and
# has no inverse, so 99 of 100 is as deep as inversion goes. 50 matches where the ddpm and sdedit
# baselines sit, so the three methods can be compared at one operating point.
LORA_TSTART=(50 99)
LORA_CFG_TAR=(3.5)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT=full

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

# Matches the existing baselines' naming so the tables line up:
# stableaudio_odeinv_cfgsrc1.0_cfgtar3.5_t99_s100
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  echo "stableaudio_odeinv_cfgsrc${LORA_CFG_SRC}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
