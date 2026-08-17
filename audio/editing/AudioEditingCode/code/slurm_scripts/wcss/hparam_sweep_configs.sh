# ABOUTME: The AudioLDM2 editing hyperparameter grid (method x tstart x cfg_tar) and the run-name
# ABOUTME: rule, sourced by both the edit job and the eval job so they cannot disagree.

# A single operating point cannot rank the three methods: one preserves better, another follows
# the caption better. Sweeping the two knobs that trade those off traces a curve per method, and
# the curves are what compare. tstart is the shared strength knob -- the fraction of the
# trajectory that gets regenerated -- and it means the same thing in all three methods:
#   ddpm    inverts all 200 steps, then reverses the last tstart of them
#   ddim    inverts tstart steps, then denoises the same tstart steps
#   sdedit  noises the input to the tstart-th timestep, then denoises from there
# tstart=200 therefore means "regenerate everything" and tstart=50 "touch only the last quarter".
# cfg_src stays at the published 3.0: at cfg_src 1.0 vs 3.0 the DDIM baseline moved 0.068 dB, so
# it does not buy a dimension worth 16 more runs.
SWEEP_MODES=(ddpm ddim sdedit)
SWEEP_TSTART=(50 100 150 200)
SWEEP_CFG_TAR=(3.0 6.0 12.0 18.0)
SWEEP_STEPS=200
SWEEP_CFG_SRC=3.0
SWEEP_SPLIT=hparam

# Emits "mode|tstart|cfg_tar" per line. The array index is the line number, so this ordering is
# what maps a SLURM task to a config -- never reorder it while a sweep is in flight.
sweep_configs() {
  local mode tstart cfg
  for mode in "${SWEEP_MODES[@]}"; do
    for tstart in "${SWEEP_TSTART[@]}"; do
      for cfg in "${SWEEP_CFG_TAR[@]}"; do
        echo "$mode|$tstart|$cfg"
      done
    done
  done
}

# Names follow the existing baseline runs (audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200) with
# the split inserted, so a sweep run can never be confused with a 696-row one. SDEdit has no
# inversion pass, so no cfg_src in its name.
sweep_run_name() {
  local mode="${1:?mode}" tstart="${2:?tstart}" cfg="${3:?cfg_tar}"
  if [ "$mode" = "sdedit" ]; then
    echo "audioldm2_sdedit_${SWEEP_SPLIT}_cfgtar${cfg}_t${tstart}_s${SWEEP_STEPS}"
  else
    echo "audioldm2_${mode}_${SWEEP_SPLIT}_cfgsrc${SWEEP_CFG_SRC}_cfgtar${cfg}_t${tstart}_s${SWEEP_STEPS}"
  fi
}
