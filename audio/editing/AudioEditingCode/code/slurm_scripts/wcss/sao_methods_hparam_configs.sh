# ABOUTME: DDPM-inversion and SDEdit on the same hparam grid as the odeinv arms, so the adapter's
# ABOUTME: effect can be read against the gap between two genuinely different editing methods.

# Pass the method at submit time: METHOD=ddpm or METHOD=sdedit. Both use Stable Audio's native
# cosine scheduler, so unlike the rejected `ddim` path their numbers were never in question -- but
# they only existed at one full-split operating point, which cannot be plotted against a 115-row
# front. This runs them over the odeinv grid instead.
LORA_MODE="${METHOD:?set METHOD=ddpm or METHOD=sdedit}"
LORA_CHECKPOINTS=("")  # no adapter: these are reference methods, not arms

LORA_TSTART=(25 50 75 99)
LORA_CFG_TAR=(3.5 7.0)
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

# stableaudio_ddpm_hparam_cfgtar3.5_t50_s100, matching the odeinv naming so the plotter and the
# eval job derive the same directories.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  echo "stableaudio_${LORA_MODE}_${LORA_SPLIT}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
