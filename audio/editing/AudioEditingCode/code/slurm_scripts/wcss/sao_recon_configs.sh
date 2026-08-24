# ABOUTME: The Stable Audio reconstruction ladder: the editing pipeline with the target caption
# ABOUTME: replaced by the source one and cfg_tar=1.0, over the whole checkpoint ladder.

# Reconstruction isolates how exactly inversion round-trips real audio, which is what the adapter
# is trained to improve -- unlike the edit benchmark, where the reverse pass runs a different
# prompt under guidance. Every step_* checkpoint is scored so a dose-response curve exists: the
# AudioLDM2 run improved reconstruction and still moved no editing metric, and the absence of any
# dose-response was one of the four legs that result rests on.
# Every save, not a subsample: an arm is 35 tracks and ~10 minutes, so the whole ladder is
# cheaper than one editing cell, and the shape of the curve is the point.
LORA_CHECKPOINTS=(
  ""  # the frozen teacher
  "sao_r8_a4_lr5e-5/checkpoint_step_2000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_4000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_6000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_8000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_10000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_12000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_14000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_16000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_18000.pt"
  "sao_r8_a4_lr5e-5/checkpoint_step_18000_ema.pt"
)

# One cell per checkpoint: full inversion (tstart = steps) and cfg_tar=1.0, so the output should
# be the input. 35 distinct MedleyDB tracks rather than every row, since a reconstruction depends
# only on (audio, source caption) and the benchmark repeats each track ~6.5 times.
LORA_TSTART=(100)
LORA_CFG_TAR=(1.0)
LORA_STEPS=100
LORA_CFG_SRC=1.0
LORA_SPLIT=full
LORA_EXTRA_ARGS=(--reconstruct True --unique_tracks True)

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

# stableaudio_recon_tracks_s100_<checkpoint>, matching the AudioLDM2 recon_tracks_* naming so
# compare_reconstruction.py reads both without special cases.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}"
  if [ -z "$ckpt" ]; then
    echo "stableaudio_recon_tracks_s${LORA_STEPS}_nolora"
    return
  fi
  echo "stableaudio_recon_tracks_s${LORA_STEPS}_$(basename "$ckpt" .pt)"
}
