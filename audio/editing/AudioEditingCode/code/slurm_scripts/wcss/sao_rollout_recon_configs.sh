# ABOUTME: Reconstruction gate for the multi-step rollout adapters: no-LoRA, the one-step adapter
# ABOUTME: they were fine-tuned from, and rollout k=4 / k=8, on real (MedleyDB) or generated audio.
#
# SPLIT=full       35 distinct MedleyDB tracks (real audio, the benchmark's inputs)
# SPLIT=genhparam  75 generated clips from the adapter's own training distribution, where exact
#                  inversion is attainable, so the gap to it is the compounding error left.
# Same operating point as the earlier reconstruction gates: source caption, cfg 1, t99 of 100.
LORA_MODE=odeinv
LORA_CHECKPOINTS=(
  ""                                                  # frozen teacher
  "saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt"       # one-step adapter (the rollout runs' init)
  "saocos_rollout4_r8_a4_lr5e-5/checkpoint_step_1000.pt"
  "saocos_rollout8_r8_a4_lr5e-5/checkpoint_step_1000.pt"
)
LORA_TSTART=(99)
LORA_CFG_TAR=(1.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT="${SPLIT:-full}"
case "$LORA_SPLIT" in
  full) LORA_EXTRA_ARGS=(--reconstruct True --unique_tracks True) ;;
  genhparam) LORA_EXTRA_ARGS=(--reconstruct True) ;;
  *) echo "SPLIT must be full or genhparam, got $LORA_SPLIT" >&2; return 1 ;;
esac
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

# stableaudio_rollrecon_<split>_s100_nolora, or ..._<run>_<checkpoint stem>
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  local tail="rollrecon_${LORA_SPLIT}_s${LORA_STEPS}"
  if [ -z "$ckpt" ]; then
    echo "stableaudio_${tail}_nolora"
  else
    echo "stableaudio_${tail}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)"
  fi
}
