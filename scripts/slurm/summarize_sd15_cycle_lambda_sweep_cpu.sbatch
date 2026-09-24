#!/bin/bash
#SBATCH --job-name=cycle-lambda-tex
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
cd "${REPO_DIR}"
module load python/3.11.15

python3 -m diff_inversion.eval.summarize_cycle_lambda_sweep \
  --output reports/sd15_cycle_lambda_sweep_r16_results.tex
