#!/bin/bash -l
# ABOUTME: PWR/WCSS array job for the noise-normality vs reconstruction benchmark: one task per
# ABOUTME: shard of the 1024 clips, every arm (DDIM, DDIM+LoRA, DDPM, Gaussian) per task.
#SBATCH --job-name=noise-recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-15
#SBATCH --output=outputs/logs/slurm/noise-recon-%A_%a.out
#SBATCH --error=outputs/logs/slurm/noise-recon-%A_%a.err
#
# Submit from audio/ with MODEL=audioldm2 or MODEL=stable_audio; the array range must match
# num_shards in config/noise_recon_$MODEL.yaml. A finished shard is skipped on resubmission.
# RUN_NAME (default main) picks outputs/noise_recon/$MODEL/$RUN_NAME, so a rerun keeps the old one.
#   sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION --export=ALL,MODEL=audioldm2 \
#     editing/AudioEditingCode/code/slurm_scripts/wcss/run_noise_recon.sh

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a
: "${MODEL:?set MODEL=audioldm2 or MODEL=stable_audio}"

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

echo "node=$(hostname) model=$MODEL run=${RUN_NAME:-main} shard=$SLURM_ARRAY_TASK_ID git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/inversion_lora/noise_recon_benchmark.py --config-name "noise_recon_$MODEL" \
  device=cuda:0 shard_id="$SLURM_ARRAY_TASK_ID" run_dir="outputs/noise_recon/$MODEL/${RUN_NAME:-main}" \
  || exit 1
echo "done"
