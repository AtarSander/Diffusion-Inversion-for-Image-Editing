# ABOUTME: Edit sweep for the dense-dataset adapter (H2): the saocos_dense991 checkpoint over the
# ABOUTME: standard odeinv grid, paired against no-LoRA twins and the coarse-dataset adapter
# ABOUTME: already scored at identical settings.

LORA_MODE=odeinv
# Written by run_train_inversion_lora.sh preset 36 (RUN_PREFIX=saocos_). No no-LoRA entry: the
# odeinv_nolora twins at these cells exist from the sao_lora_sweep_configs.sh runs.
LORA_CHECKPOINTS=(
  "saocos_dense991_r8_a4_lr5e-5/checkpoint_step_3000.pt"
)

LORA_TSTART=(25 50 75 99)
LORA_CFG_TAR=($(tr ":," "  " <<< "${CFG_TARS:-3.5 7.0}"))
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT="${SPLIT:-hparam}"

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

# stableaudio_odeinvlora_hparam_saocos_dense991_r8_a4_lr5e-5_checkpoint_step_3000_cfgtar<c>_t<t>_s100,
# the same shape as the other adapter arms so the eval and the plotter derive the directories.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  local config="$(dirname "$ckpt")"
  local stem="$(basename "$ckpt" .pt)"
  echo "stableaudio_odeinvlora_${LORA_SPLIT}_${config}_${stem}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}

# The Stable Audio driver appends dataset_name to a path already containing it, so its runs
# live under medleymd/stable_audio relative to the edits root. Owned here so an edit or eval
# submission cannot pair this grid with the wrong directory.
LORA_EDITS_SUBDIR="${LORA_EDITS_SUBDIR:-medleymd/stable_audio}"
