# ABOUTME: AudioLDM2 matched-NFE editing grid: every method at ~BUDGET denoiser calls, tracing
# ABOUTME: its front by inversion depth. DDIM carries the three adapter arms (no-LoRA / LoRA-Gen /
# ABOUTME: LoRA-Real); DDPM and SDEdit are the reference methods.

# Per-edit NFE, DERIVED from the edit code (get_noise_pred = 2 forwards, guided both passes) and
# to be CONFIRMED by run_nfe_smoke_audioldm2.sh before the full launch:
#   ddim    4T           -- 2T guided inversion (T=tstart steps) + 2T guided denoise; N is free
#   ddpm    2N + 2T      -- 2N guided forward process (whole grid) + 2T guided reverse
#   sdedit  2T           -- no inversion; 2T guided denoise
# At BUDGET=800 (the benchmark's N=200/T=200 DDIM cost), depth = T/N is the free front axis.
BUDGET=800
LORA_MODE="${METHOD:?set METHOD=ddim, ddpm or sdedit}"
LORA_CFG_TAR=($(tr ":," "  " <<< "${CFG_TARS:-6.0 12.0}"))
LORA_CFG_SRC=3.0
LORA_SPLIT="${SPLIT:-hparam}"
LORA_STEPS=200          # per-row steps override this
LORA_EDITS_SUBDIR="${LORA_EDITS_SUBDIR:-audioldm2_ddim}"

# depth% : tstart : steps, solved per method for BUDGET=800.
case "$LORA_MODE" in
  ddim)   POINTS=(100:200:200 50:200:400 25:200:800) ;;   # 4T=800, N free
  ddpm)   POINTS=(100:200:200 50:133:267 25:80:320) ;;     # 2N+2T=800
  sdedit) POINTS=(100:400:400 50:400:800) ;;               # 2T=800, N free (25% -> N1600 dropped)
  *) echo "unknown METHOD=$LORA_MODE" >&2; return 1 ;;
esac

# DDIM is the only pass an adapter replaces, so only it carries the LoRA arms. "" is no-LoRA.
if [ "$LORA_MODE" = "ddim" ]; then
  LORA_CHECKPOINTS=(
    ""
    "attn_r8_a4_lr5e-5/checkpoint_step_4000.pt"          # LoRA-Gen (trajectory-trained)
    "aldm2realfn_r8_a4_lr5e-5/checkpoint_step_3000.pt"   # LoRA-Real (forward-noise real audio)
  )
else
  LORA_CHECKPOINTS=("")
fi

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

# audioldm2_<mode>_nolora_hparam_nfe800_t<T>_s<N>_cfgtar<c>, or ..._<mode>lora_<run>_<stem>_...
lora_sweep_run_name() {
  local ckpt="${1?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}" steps="${4:?steps}"
  local tail="hparam_nfe${BUDGET}_t${tstart}_s${steps}_cfgtar${cfg}"
  if [ -z "$ckpt" ]; then
    echo "audioldm2_${LORA_MODE}_nolora_${tail}"
  else
    echo "audioldm2_${LORA_MODE}lora_$(dirname "$ckpt")_$(basename "$ckpt" .pt)_${tail}"
  fi
}
