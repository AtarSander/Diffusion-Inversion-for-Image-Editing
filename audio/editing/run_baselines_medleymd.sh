#!/bin/bash
# ABOUTME: Reproduce the AudioLDM2 and Stable Audio editing baselines (DDPM-inv, DDIM-inv,
# ABOUTME: SDEdit) on the full 696-row MedleyMD benchmark, sharded across the free GPUs.
#
# Each GPU runs one shard of every config, so the six configs finish at roughly the same time
# and any single shard can be re-run independently. Shards of a config share a fixed run_name
# so they write into one output directory.
#
# Usage:  bash editing/run_baselines_medleymd.sh [gpu_ids...]      (default: 3 4 5 6 7)
# Logs:   audio/outputs/logs/baselines_<stamp>/gpu<N>.log

set -uo pipefail

AUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$AUDIO_ROOT/editing/AudioEditingCode/code"
PYTHON="$AUDIO_ROOT/.venv/bin/python"

GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then GPUS=(3 4 5 6 7); fi
N_PARTS=${#GPUS[@]}

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$AUDIO_ROOT/outputs/logs/baselines_${STAMP}"
mkdir -p "$LOG_DIR"

# script|mode|steps|cfg_src|cfg_tar|tstart|run_name
# DDIM uses tstart == steps: partial inversion is not DDIM inversion, and this is the baseline
# our LoRA inversion is meant to improve. The README states the same requirement.
CONFIGS=(
  "edit_audioldm_medleydb.py|ddpm|200|3.0|12.0|100|audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200"
  "edit_audioldm_medleydb.py|ddim|200|3.0|12.0|200|audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200"
  "edit_audioldm_medleydb.py|sdedit|200|3.0|12.0|100|audioldm2_sdedit_cfgtar12.0_t100_s200"
  "edit_stableaudio_medleydb.py|ddpm|100|1.0|3.5|50|stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100"
  "edit_stableaudio_medleydb.py|ddim|100|1.0|3.5|100|stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100"
  "edit_stableaudio_medleydb.py|sdedit|100|1.0|3.5|50|stableaudio_sdedit_cfgtar3.5_t50_s100"
)

echo "Launching ${#CONFIGS[@]} configs x ${N_PARTS} shards on GPUs: ${GPUS[*]}"
echo "Logs: $LOG_DIR"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  part="$i"
  log="$LOG_DIR/gpu${gpu}.log"

  # setsid so the run survives this shell going away; it is a local job, not SLURM.
  setsid bash -c "
    cd '$CODE_DIR' || exit 1
    echo \"[\$(date)] worker start: gpu=$gpu part=$part/$N_PARTS\"
    failures=0
    for cfg in ${CONFIGS[*]@Q}; do
      IFS='|' read -r script mode steps cfg_src cfg_tar tstart run_name <<< \"\$cfg\"
      echo \"[\$(date)] ==== \$run_name (shard $part/$N_PARTS) ====\"
      CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' \"\$script\" \\
        --mode \"\$mode\" \\
        --num_diffusion_steps \"\$steps\" \\
        --cfg_src \"\$cfg_src\" \\
        --cfg_tar \"\$cfg_tar\" \\
        --tstart \"\$tstart\" \\
        --seed 42 \\
        --n_parts $N_PARTS \\
        --part_id $part \\
        --run_name \"\$run_name\"
      rc=\$?
      if [ \$rc -ne 0 ]; then
        echo \"[\$(date)] FAILED rc=\$rc : \$run_name shard $part\"
        failures=\$((failures+1))
      else
        echo \"[\$(date)] done: \$run_name shard $part\"
      fi
    done
    echo \"[\$(date)] worker finished: gpu=$gpu part=$part failures=\$failures\"
  " > "$log" 2>&1 < /dev/null &

  echo "  gpu $gpu -> shard $part -> $log"
done

echo
echo "Watch:  tail -f $LOG_DIR/gpu*.log"
echo "Count:  find $AUDIO_ROOT/outputs/edits -name 'a*.wav' | wc -l   (target: 6 x 696 = 4176)"
