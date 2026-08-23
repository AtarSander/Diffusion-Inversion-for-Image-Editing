# ABOUTME: The Stable Audio arm of the inversion-LoRA editing sweep: which checkpoints, over which
# ABOUTME: DPMSolver grid, sourced by both the edit and the eval job so they cannot disagree.

# Stable Audio's DDIM mode is the only invertible one (its native cosine scheduler is an unseeded
# SDE), so this arm is DDIM-only. Unlike the AudioLDM2 sweep, the no-LoRA twins do not exist yet --
# only three full-split baselines do -- so the empty checkpoint entry below runs them here, paired
# over the same 115 rows at identical settings.
LORA_CHECKPOINTS=(
  ""  # no adapter: the paired reference arm
  "sao_r8_a4_lr5e-5/checkpoint_step_18000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_18000_ema.pt"
)

# tstart traces the front. Stable Audio's baselines run a 100-step grid at cfg_src 1.0, so the
# tstart=100 / cfg_tar=3.5 cell reproduces the run already on disk and doubles as a sanity check.
LORA_TSTART=(25 50 75 100)
LORA_CFG_TAR=(3.5 7.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT=hparam

# Emits "checkpoint|tstart|cfg_tar" per line; the array index is the line number, so this
# ordering must not change while a sweep is in flight.
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

# stableaudio_ddimlora_hparam_<training run>_<checkpoint>_cfgtar<c>_t<t>_s100, or
# stableaudio_ddim_nolora_hparam_... for the reference arm, so a run directory names both the
# adapter and the operating point it was measured at.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  if [ -z "$ckpt" ]; then
    echo "stableaudio_ddim_nolora_${LORA_SPLIT}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
    return
  fi
  local config="$(dirname "$ckpt")"
  local stem="$(basename "$ckpt" .pt)"
  echo "stableaudio_ddimlora_${LORA_SPLIT}_${config}_${stem}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
