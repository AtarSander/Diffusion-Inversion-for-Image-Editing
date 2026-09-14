#!/bin/bash -l
# ABOUTME: PWR/WCSS single-task job generating the genhparam eval inputs (75 Stable Audio clips
# ABOUTME: from the source captions) and their paired lower-bound reference set.
#SBATCH --job-name=lorainv-geninputs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/geninputs-%j.out
#SBATCH --error=outputs/logs/slurm/geninputs-%j.err
#
# Submit from audio/ (submit_lora_sweep.sh-style wrapper is overkill for one task):
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_generate_eval_inputs.sh

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/inversion_lora/generate_eval_inputs.py device=cuda:0 || exit 1

# The paired reference for PSNR/SSIM and FAD is a copy of each row's source clip; build it here
# so the eval jobs chained on this one find it complete.
python -m editing.build_lower_bound --splits genhparam || exit 1

echo "done"
