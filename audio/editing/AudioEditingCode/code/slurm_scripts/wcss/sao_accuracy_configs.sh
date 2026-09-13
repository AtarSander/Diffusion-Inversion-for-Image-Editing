# ABOUTME: Experiment 2 -- is inversion accuracy the binding constraint on editing? The same
# ABOUTME: checkpoint ladder scored twice: reconstruction (accuracy) and editing (what we care about).

# The adapter closes 90.7% of the shift gap and moves LPAPS by 0.08. If editing flattens while
# reconstruction keeps improving, more inversion accuracy cannot help and experiments that buy
# accuracy (CFG-aware training, second order) are capped before they start. Run this first.
#
# PROBE=recon  -- source caption, cfg_tar 1.0, full inversion: output should be the input.
# PROBE=edit   -- the real benchmark at two operating points, so each checkpoint has a mini-front
#                 and "LPAPS at matched CLAP" is answerable rather than confounded by CLAP drift.
PROBE="${PROBE:?set PROBE=recon or edit}"

# odeinv, never ddim: the ddim sampler is rejected (output/sao_schedules/REPORT.md) and the old
# sao_recon_configs.sh omits LORA_MODE entirely, so it silently ran the wrong one.
LORA_MODE=odeinv

# The corrected cosine-ODE run, which never got a reconstruction ladder -- the existing one on
# disk belongs to the rejected sao_r8 ddim run.
LORA_CHECKPOINTS=(
  ""  # the frozen teacher: the zero-accuracy end of the ladder
  "saocos_r8_a4_lr5e-5/checkpoint_step_2000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_6000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_8000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_10000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_12000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_14000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_16000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_18000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_20000.pt"
  "saocos_r8_a4_lr5e-5/checkpoint_step_20000_ema.pt"
)

LORA_STEPS=100
LORA_CFG_SRC=1.0
case "$PROBE" in
  recon)
    # 99, not 100: on odeinv the last reverse step ends at sigma = 0 and has no inverse, so the
    # edit script asserts steps in (0, 99]. The old sao_recon_configs.sh uses 100 because it ran
    # the ddim mode, where that step exists.
    LORA_TSTART=(99)
    LORA_CFG_TAR=(1.0)
    LORA_SPLIT=full
    # 35 distinct tracks, not all 115 rows: a reconstruction depends only on (audio, source
    # caption) and the benchmark repeats each track ~6.5 times.
    LORA_EXTRA_ARGS=(--reconstruct True --unique_tracks True)
    ;;
  edit)
    LORA_TSTART=(50 75)
    LORA_CFG_TAR=(3.5)
    LORA_SPLIT=hparam
    LORA_EXTRA_ARGS=()
    ;;
  *) echo "unknown PROBE=$PROBE" >&2; return 1 ;;
esac

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

# The training run is part of the name. sao_recon_configs.sh used only the checkpoint basename,
# so a saocos ladder would have overwritten the sao_r8 ladder already on disk, step for step.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  local tail
  if [ "$PROBE" = "recon" ]; then tail="recon_tracks_s${LORA_STEPS}"
  else tail="edit_hparam_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"; fi
  if [ -z "$ckpt" ]; then
    echo "stableaudio_acc_${tail}_nolora"
    return
  fi
  echo "stableaudio_acc_${tail}_$(dirname "$ckpt")_$(basename "$ckpt" .pt)"
}
