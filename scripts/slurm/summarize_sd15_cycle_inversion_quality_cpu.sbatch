#!/bin/bash
#SBATCH --job-name=sd15-invqual-summary
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
#SBATCH --error=logs/slurm/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=szymczyk.alek1@gmail.com

set -euo pipefail

REPO_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
RESULTS_ROOT=$REPO_DIR/reports/eval/sd15_cycle_inversion_quality

for variant in ddim standard_r8 standard_r16 cycle_r8 cycle_r16 cycle_r32; do
  test -s "$RESULTS_ROOT/$variant/evaluation_summary.json"
done

cd "$REPO_DIR"
python3 -m scripts.analysis.summarize_sd15_cycle_inversion_quality
