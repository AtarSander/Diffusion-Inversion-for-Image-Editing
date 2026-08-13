#!/bin/bash -l
# ABOUTME: PWR/WCSS job that builds the training venv and pre-fetches model weights, so nothing
# ABOUTME: has to be installed or downloaded from the login node.
#SBATCH --job-name=lorainv-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=outputs/logs/slurm/setup-%j.out
#SBATCH --error=outputs/logs/slurm/setup-%j.err
#
# Submit from audio/:
#   bash editing/AudioEditingCode/code/slurm_scripts/wcss/submit_setup.sh
#
# The GPU is not needed for the install itself; it is requested only so the job lands on the
# same known-good account/partition combination as everything else.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # expected: <repo>/audio
set -a; [ -f .env ] && source .env; set +a

module load Python/3.10.4-GCCcore-11.3.0

if [ ! -d .venv ]; then
  echo "creating .venv"
  python -m venv .venv || exit 1
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_lorainv.txt
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: pip install" >&2
  exit "$rc"
fi

export PYTHONPATH="$PWD:$PWD/editing/AudioEditingCode/code:${PYTHONPATH:-}"

# Every later array task would otherwise pull the same ~12 GB concurrently.
python editing/AudioEditingCode/code/slurm_scripts/wcss/prefetch_models.py || exit 1

python - <<'PY'
import importlib
for module in ("torch", "diffusers", "transformers", "peft", "skimage", "wandb", "hydra"):
    importlib.import_module(module)
    print(f"import {module}: ok")
PY
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc: import verification" >&2
  exit "$rc"
fi
echo "done: environment ready"
