# ABOUTME: Real-audio reconstruction, four arms: DDPM-inv (exact by construction — the reference),
# ABOUTME: no-LoRA ODEInv, the trajectory adapter, and the real-audio (forward-noise) adapter.
# ABOUTME: The realfn arm is the chain-consistency gate: if its multi-step inversion composes,
# ABOUTME: reconstruction must improve markedly over no-LoRA; if not, the chainless pairs do not
# ABOUTME: compose and the realfn editing numbers are partly artifact.

# METHOD=ddpm runs the DDPM reference arm; default is the three odeinv arms. Names match the
# existing acc_recon ladder so SKIP_EXISTING=1 reuses the no-LoRA and trajectory-adapter edits
# already on disk and only the new arms actually edit.
LORA_MODE="${METHOD:-odeinv}"
if [ "$LORA_MODE" = "odeinv" ]; then
  LORA_CHECKPOINTS=(
    ""                                                    # exists: ..._nolora
    "saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt"         # exists: acc_recon ladder
    "saocos_realfn_r8_a4_lr5e-5/checkpoint_step_3000.pt"  # new
  )
else
  LORA_CHECKPOINTS=("")
fi

# Reconstruction settings: full invertible depth, denoise with the SOURCE caption at cfg 1.0,
# one row per distinct MedleyDB track (a reconstruction depends only on the audio and caption).
LORA_TSTART=(99)
LORA_CFG_TAR=(1.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT=full
LORA_EXTRA_ARGS=(--reconstruct True --unique_tracks True)

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
  local tail="recon_tracks_s${LORA_STEPS}"
  if [ "$LORA_MODE" = "ddpm" ]; then
    echo "stableaudio_acc_${tail}_ddpm"
  elif [ -z "$ckpt" ]; then
    echo "stableaudio_acc_${tail}_nolora"
  else
    echo "stableaudio_acc_${tail}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)"
  fi
}
