# ABOUTME: Every method at one fixed NFE budget, tracing its front by inversion depth instead of by
# ABOUTME: tstart, so the comparison is at equal compute rather than equal nominal tstart.

# Measured denoiser calls per edit (verified by an NFE counter in StableAudWrapper.unet_forward,
# at N=20/T=10: 32, 60, 20 respectively):
#   odeinv  3T + 2      -- T single-branch inversion calls at cfg_src=1.0, 2(T+1) guided reverse
#   ddpm    2N + 2T     -- its forward process runs the whole grid regardless of tstart
#   sdedit  2T          -- no inversion pass at all
# Fixing the budget therefore fixes T per method, leaving N free; N sets how deep in sigma the
# inversion reaches, so depth = T/N becomes the front axis. The adapter is merged into the base
# weights, so ours costs no more than the no-LoRA arm.
BUDGET=300
LORA_MODE="${METHOD:?set METHOD=odeinv, ddpm or sdedit}"
LORA_CFG_TAR=(3.5)
LORA_CFG_SRC=1.0
LORA_SPLIT=hparam
LORA_STEPS=100  # per-row steps override this

# depth:tstart:steps, solved per method for BUDGET
case "$LORA_MODE" in
  odeinv) POINTS=(25:99:396 50:99:198 75:99:132 100:99:100) ;;   # 3T+2 = 299
  ddpm)   POINTS=(25:30:120 50:50:100 75:64:86 100:75:75) ;;      # 2N+2T = 300
  sdedit) POINTS=(25:150:600 50:150:300 75:150:200 100:150:150) ;; # 2T = 300
  *) echo "unknown METHOD=$LORA_MODE" >&2; return 1 ;;
esac

# Only the odeinv arm has an adapter to test; the reference methods run without one.
if [ "$LORA_MODE" = "odeinv" ]; then
  LORA_CHECKPOINTS=("" "saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt")
else
  LORA_CHECKPOINTS=("")
fi

# Emits "checkpoint|tstart|cfg_tar|steps".
lora_sweep_configs() {
  local ckpt point cfg depth tstart steps
  for ckpt in "${LORA_CHECKPOINTS[@]}"; do
    for point in "${POINTS[@]}"; do
      IFS=':' read -r depth tstart steps <<< "$point"
      for cfg in "${LORA_CFG_TAR[@]}"; do
        echo "$ckpt|$tstart|$cfg|$steps"
      done
    done
  done
}

# stableaudio_odeinv_nolora_hparam_nfe300_t99_s198_cfgtar3.5, or ..._odeinvlora_hparam_<run>_<stem>_...
# steps must come in as an argument: at a fixed budget tstart is constant across depths for
# odeinv and sdedit, so deriving it from tstart collapsed all four points onto one directory.
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}" steps="${4:?steps}"
  local tail="hparam_nfe${BUDGET}_t${tstart}_s${steps}_cfgtar${cfg}"
  if [ -z "$ckpt" ]; then
    echo "stableaudio_${LORA_MODE}_nolora_${tail}"
  else
    echo "stableaudio_${LORA_MODE}lora_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_${tail}"
  fi
}
