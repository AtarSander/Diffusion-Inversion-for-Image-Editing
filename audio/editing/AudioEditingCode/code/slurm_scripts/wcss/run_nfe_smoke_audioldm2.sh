#!/bin/bash -l
# ABOUTME: Measure AudioLDM2 per-edit NFE for ddim/ddpm/sdedit at a few (steps, tstart) points,
# ABOUTME: so the matched-NFE grid formulas are fit from real counts, not guessed.
#SBATCH --job-name=lorainv-nfesmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/nfesmoke-%j.out
#SBATCH --error=outputs/logs/slurm/nfesmoke-%j.err
#
# Submit from audio/:
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_nfe_smoke_audioldm2.sh

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
set -a; [ -f .env ] && source .env; set +a
module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"
cd editing/AudioEditingCode/code

# One edit per (mode, steps, tstart); part_id 0 of n_parts=115 is a single row. cfg_src 3.0 /
# cfg_tar 12.0 are the benchmark guidances (guided both passes), so the counts match deployment.
for mode in ddim ddpm sdedit; do
  for pt in "40 10" "40 20" "80 20"; do
    read -r N T <<< "$pt"
    echo "=== MEASURE mode=$mode steps=$N tstart=$T ==="
    python edit_audioldm_medleydb.py --mode "$mode" --num_diffusion_steps "$N" --tstart "$T" \
      --cfg_src 3.0 --cfg_tar 12.0 --split hparam --n_parts 115 --part_id 0 \
      --run_name "nfesmoke_${mode}_s${N}_t${T}" 2>&1 | grep -E "^NFE:" || echo "  (no NFE line)"
  done
done
echo "SMOKE DONE"
