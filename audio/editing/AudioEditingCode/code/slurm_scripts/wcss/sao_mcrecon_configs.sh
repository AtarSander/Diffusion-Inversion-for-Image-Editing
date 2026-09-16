# ABOUTME: MusicCaps reconstruction (mcrecon split), four arms: DDPM reference, no-LoRA ODEInv,
# ABOUTME: the trajectory adapter and the realfn adapter — the training-audio-source control for
# ABOUTME: the MedleyDB recon4 comparison.

# METHOD=ddpm runs the DDPM reference arm; default is the three odeinv arms.
LORA_MODE="${METHOD:-odeinv}"
if [ "$LORA_MODE" = "odeinv" ]; then
  LORA_CHECKPOINTS=(
    ""
    "saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt"
    "saocos_realfn_r8_a4_lr5e-5/checkpoint_step_3000.pt"
  )
else
  LORA_CHECKPOINTS=("")
fi

# Full invertible depth, denoise with the source caption at cfg 1.0 (reconstruction).
LORA_TSTART=(99)
LORA_CFG_TAR=(1.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT=mcrecon
LORA_EXTRA_ARGS=(--reconstruct True)

LORA_EDITS_SUBDIR="${LORA_EDITS_SUBDIR:-medleymd/stable_audio}"

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
  local tail="recon_mcrecon_s${LORA_STEPS}"
  if [ "$LORA_MODE" = "ddpm" ]; then
    echo "stableaudio_acc_${tail}_ddpm"
  elif [ -z "$ckpt" ]; then
    echo "stableaudio_acc_${tail}_nolora"
  else
    echo "stableaudio_acc_${tail}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)"
  fi
}
