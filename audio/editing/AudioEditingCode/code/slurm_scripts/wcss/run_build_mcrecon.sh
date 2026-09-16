#!/bin/bash -l
# ABOUTME: PWR/WCSS CPU job building the mcrecon split: MusicCaps clips relaid as mc{idx}_MIX.wav,
# ABOUTME: the prompt CSV, and the paired lower-bound reference. Run before the recon sweeps.
#SBATCH --job-name=lorainv-mcrecon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/mcrecon-%j.out
#SBATCH --error=outputs/logs/slurm/mcrecon-%j.err
#
# Submit from audio/:
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_build_mcrecon.sh
# NUM overrides how many leading caption rows to consider (default 200).

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

python src/inversion_lora/build_musiccaps_recon.py --num "${NUM:-200}" || exit 1
echo "done"
