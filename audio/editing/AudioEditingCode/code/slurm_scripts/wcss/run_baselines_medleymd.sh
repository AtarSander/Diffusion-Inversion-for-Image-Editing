#!/bin/bash
# ABOUTME: Submit the six MedleyMD editing baselines (DDPM-inv, DDIM-inv, SDEdit x AudioLDM2,
# ABOUTME: Stable Audio) to SLURM on WCSS, one job per (config, shard).
#
# Prerequisites on the cluster, once:
#   git pull
#   cd audio && uv venv --python 3.11 .venv
#   uv pip install --python .venv/bin/python -r requirements_lorainv.txt
#   cat > .env <<'EOF'
#   MEDLEYDB_AUDIO_DIR=/path/to/MedleyDB/V1_mix      # <Track>/<Track>_MIX.wav layout
#   HF_TOKEN=hf_...                                  # Stable Audio Open is license-gated
#   EOF
#
# Then:  bash slurm_scripts/wcss/run_baselines_medleymd.sh
# Dry run (print sbatch scripts without submitting): SUBMIT=0 bash .../run_baselines_medleymd.sh

set -uo pipefail

# --- cluster settings: confirm these match your WCSS allocation -------------------------------
ACCOUNT="${GRANT_ACCOUNT:?Set GRANT_ACCOUNT (SLURM account)}"
PARTITION="${GRANT_PARTITION:?Set GRANT_PARTITION (e.g. a GPU partition)}"
N_CPUS="${N_CPUS:-8}"
MEM_GB="${MEM_GB:-48}"
SUBMIT="${SUBMIT:-1}"

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIO_ROOT="$(cd "$CODE_DIR/../../.." && pwd)"
PYTHON="$AUDIO_ROOT/.venv/bin/python"

# 12 shards keeps the slowest job (AudioLDM2, ~58 edits at ~280 s) near 4.5 h, comfortably
# inside an 8 h limit. Measured locally: AudioLDM2 ~280 s/edit, Stable Audio ~94 s/edit.
N_PARTS="${N_PARTS:-12}"

LOG_DIR="$AUDIO_ROOT/outputs/logs/slurm/baselines"
mkdir -p "$LOG_DIR"

# script|mode|steps|cfg_src|cfg_tar|tstart|walltime|run_name
# DDIM uses tstart == steps: partial inversion is not DDIM inversion, and this is the exact
# baseline the LoRA inversion is meant to improve.
CONFIGS=(
  "edit_audioldm_medleydb.py|ddpm|200|3.0|12.0|100|08:00:00|audioldm2_ddpm_cfgsrc3.0_cfgtar12.0_t100_s200"
  "edit_audioldm_medleydb.py|ddim|200|3.0|12.0|200|08:00:00|audioldm2_ddim_cfgsrc3.0_cfgtar12.0_t200_s200"
  "edit_audioldm_medleydb.py|sdedit|200|3.0|12.0|100|08:00:00|audioldm2_sdedit_cfgtar12.0_t100_s200"
  "edit_stableaudio_medleydb.py|ddpm|100|1.0|3.5|50|04:00:00|stableaudio_ddpm_cfgsrc1.0_cfgtar3.5_t50_s100"
  "edit_stableaudio_medleydb.py|ddim|100|1.0|3.5|100|04:00:00|stableaudio_ddim_cfgsrc1.0_cfgtar3.5_t100_s100"
  "edit_stableaudio_medleydb.py|sdedit|100|1.0|3.5|50|04:00:00|stableaudio_sdedit_cfgtar3.5_t50_s100"
)

echo "code dir : $CODE_DIR"
echo "python   : $PYTHON"
echo "shards   : $N_PARTS  -> $(( ${#CONFIGS[@]} * N_PARTS )) jobs"
echo "logs     : $LOG_DIR"
[ "$SUBMIT" = "1" ] || echo "SUBMIT=0: printing only, nothing will be queued"
echo

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r script mode steps cfg_src cfg_tar tstart walltime run_name <<< "$cfg"
  for part in $(seq 0 $((N_PARTS - 1))); do
    job_name="${run_name}_p${part}"
    submission=$(cat <<EOT
#!/bin/bash -l
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM_GB}G
#SBATCH --time=${walltime}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err

set -euo pipefail
cd "${CODE_DIR}"

# env.py reads MEDLEYDB_AUDIO_DIR and HF_TOKEN from audio/.env via load_dotenv.
"${PYTHON}" "${script}" \\
  --mode "${mode}" \\
  --num_diffusion_steps ${steps} \\
  --cfg_src ${cfg_src} \\
  --cfg_tar ${cfg_tar} \\
  --tstart ${tstart} \\
  --seed 42 \\
  --n_parts ${N_PARTS} \\
  --part_id ${part} \\
  --run_name "${run_name}"
EOT
)
    if [ "$SUBMIT" = "1" ]; then
      echo "$submission" | sbatch
    else
      echo "--- would submit: ${job_name}"
    fi
  done
done

echo
echo "Watch:  squeue -u \$USER"
echo "Verify: find ${AUDIO_ROOT}/outputs/edits -name 'a*.wav' | wc -l   # target 6 x 696 = 4176"
