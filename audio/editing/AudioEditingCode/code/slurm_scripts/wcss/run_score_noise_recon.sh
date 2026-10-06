#!/bin/bash -l
# ABOUTME: PWR/WCSS job scoring a finished noise-recon benchmark run (KL, Corr, mel MAE/PSNR/SSIM,
# ABOUTME: LPAPS) into output/noise_recon/<timestamp>_<model>/ in the eval environment.
#SBATCH --job-name=noise-recon-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/noise-recon-score-%j.out
#SBATCH --error=outputs/logs/slurm/noise-recon-score-%j.err
#
# Submit from audio/ with the same MODEL (and RUN_NAME, default main) as the benchmark array:
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION --dependency=afterok:$J \
#     --export=ALL,MODEL=audioldm2 editing/AudioEditingCode/code/slurm_scripts/wcss/run_score_noise_recon.sh

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a
: "${MODEL:?set MODEL=audioldm2 or MODEL=stable_audio}"

module load Python/3.10.4-GCCcore-11.3.0
source .venv_eval/bin/activate
# evals/utils.py imports `evals` as a top-level package, and the scorer imports `editing`/`src`.
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RUN="${RUN_NAME:-main}"
LABEL="$MODEL"; [ "$RUN" = main ] || LABEL="${MODEL}_$RUN"
echo "node=$(hostname) model=$MODEL run=$RUN git=$(git rev-parse HEAD)"
python -m editing.score_noise_recon --run_dir "outputs/noise_recon/$MODEL/$RUN" --label "$LABEL" \
  || exit 1
echo "done"
