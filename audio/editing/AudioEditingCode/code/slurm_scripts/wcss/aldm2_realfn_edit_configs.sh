# ABOUTME: AudioLDM2 editing comparison for the realfn adapter: no-LoRA, the trajectory adapter
# ABOUTME: (attn_r8_a4_lr5e-5, realfn's rank/lr twin), and realfn, over the hparam split.

# DDIM inversion is the only pass the adapter replaces, so this arm is ddim-only. The empty
# checkpoint is the paired no-LoRA baseline at identical settings.
LORA_MODE=ddim
LORA_CHECKPOINTS=(
  ""
  "attn_r8_a4_lr5e-5/checkpoint_step_8000.pt"
  "aldm2realfn_r8_a4_lr5e-5/checkpoint_step_3000.pt"
)

# tstart traces the front; cfg_tar 12.0 is the benchmark operating point. 200 is full DDIM
# inversion (where the adapter matters most); 150 gives a second front point.
LORA_TSTART=(${TSTARTS:-150 200})
LORA_CFG_TAR=(${CFG_TARS:-12.0})
LORA_STEPS=200
LORA_CFG_SRC=3.0
LORA_SPLIT="${SPLIT:-hparam}"

# AudioLDM2 edits live directly under the edits root's audioldm2_ddim subdir (no doubled path).
LORA_EDITS_SUBDIR="${LORA_EDITS_SUBDIR:-audioldm2_ddim}"

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
  local tail="${LORA_SPLIT}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
  if [ -z "$ckpt" ]; then
    echo "audioldm2_ddim_nolora_${tail}"
  else
    echo "audioldm2_ddimlora_${LORA_SPLIT}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
  fi
}
