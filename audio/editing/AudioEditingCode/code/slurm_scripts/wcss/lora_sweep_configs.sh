# ABOUTME: The inversion-LoRA arm of the hparam sweep: which checkpoints, over which DDIM grid,
# ABOUTME: sourced by both the edit and the eval job so they cannot disagree about the runs.

# The adapter only replaces the epsilon used during DDIM *inversion*, so this arm is DDIM-only
# and every run has a no-LoRA twin already scored at identical settings -- the comparison is
# paired over the same 115 rows, not curve-against-curve.
#
# Paths are relative to LORAINV_CHECKPOINT_ROOT. Each entry needs its .json sidecar next to it.
LORA_CHECKPOINTS=(
  "q4_full_r32_a16_lr5e-4/checkpoint_step_6000.pt"
  "q4_full_r32_a16_lr5e-4/checkpoint_step_6000_ema.pt"
  "attn_r32_a16_lr2e-4/checkpoint_step_4000.pt"
  "attn_r32_a16_lr2e-4/checkpoint_step_4000_ema.pt"
  "full_r8_a4_lr5e-4/checkpoint_step_2000.pt"
  "full_r8_a4_lr5e-4/checkpoint_step_2000_ema.pt"
  # Appended, never inserted: the array index is the line number, so 0-47 must keep meaning what
  # it meant while the earlier sweep ran. These two are the timestep-embedding preset (index 27),
  # the first adapter to break the 85-88% plateau -- 92.7% of the shift gap closed. The two steps
  # differ because raw and EMA weights peak at different points: raw is best at 1000 on both
  # val/loss (3.543e-6) and reconstruction (+0.62 dB, its only positive step), while the EMA at
  # 1000 is its worst point (9.692e-6, the average has not caught up) and bottoms at 3000
  # (3.408e-6). Pairing both arms at one step would have spent 8 runs on a checkpoint already
  # known to be bad.
  "q4_fullte_r32_a16_lr5e-4/checkpoint_step_1000.pt"
  "q4_fullte_r32_a16_lr5e-4/checkpoint_step_3000_ema.pt"
)

# tstart is the axis that traces the front; cfg_tar 6 and 12 bracket the region where DDIM's
# no-LoRA front actually lies. Adding 3.0 and 18.0 doubles the cost for points off the front.
LORA_TSTART=(50 100 150 200)
LORA_CFG_TAR=(6.0 12.0)
LORA_STEPS=200
LORA_CFG_SRC=3.0
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

# audioldm2_ddimlora_hparam_<training run>_<checkpoint>_cfgtar<c>_t<t>_s200, so a run directory
# names both the adapter and the operating point it was measured at.
lora_sweep_run_name() {
  local ckpt="${1:?checkpoint}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  local config="$(dirname "$ckpt")"
  local stem="$(basename "$ckpt" .pt)"
  echo "audioldm2_ddimlora_${LORA_SPLIT}_${config}_${stem}_cfgtar${cfg}_t${tstart}_s${LORA_STEPS}"
}
