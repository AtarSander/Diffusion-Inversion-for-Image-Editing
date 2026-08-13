#!/bin/bash -l
# ABOUTME: PWR/WCSS job gating the generated trajectory dataset before training reads it, so a
# ABOUTME: partial or inconsistent sample fails here rather than mid-run.
#SBATCH --job-name=lorainv-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/verify-%j.out
#SBATCH --error=outputs/logs/slurm/verify-%j.err
#
# Submit from audio/ after the trajectory array finishes:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_verify.sh
#
# The check is CPU-only; the GPU is requested only to reuse the known-good account/partition.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

DATA_DIR="${TRAJECTORY_DIR:-${LORAINV_DATA_ROOT:?not set or empty in .env}/audioldm2_trajectories_gonogo_fp32}"
EXPECTED="${TOTAL_SAMPLES:-1500}"

echo "verifying $DATA_DIR (expecting $EXPECTED samples)"
found=$(find "$DATA_DIR" -name meta.json 2>/dev/null | wc -l)
echo "samples with meta.json: $found"
if [ "$found" -ne "$EXPECTED" ]; then
  echo "ERROR: $found complete samples but $EXPECTED expected. Re-run the trajectory array; it" >&2
  echo "skips whatever already finished, so it is safe to resubmit." >&2
  exit 1
fi

python src/inversion_lora/verify_trajectories.py --root_dir "$DATA_DIR" --check_step 16
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: dataset did not pass verification" >&2
  exit "$rc"
fi
echo "done: dataset verified"
