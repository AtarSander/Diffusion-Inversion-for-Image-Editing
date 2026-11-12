#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -euo pipefail
: "${SLURM_JOB_ID:?Run only inside the authorized allocated job}"
# Slurm executes a spool copy; use the explicitly supplied worktree, not BASH_SOURCE.
repo=${2:?Pass the absolute isolated worktree path}
[[ "$repo" = /* ]]
cd -- "$repo"
repo=$PWD
[[ -f diff_inversion/steering/requirements.txt && -f scripts/run_wikiart_readers.py ]]
attempt=${1:?Pass the fresh attempt artifact directory}
[[ "$attempt" = /* && -f "$attempt/preparation.json" && ! -e "$attempt/run" && ! -e "$attempt/comparison" && ! -e "$attempt/SUCCESS.json" ]]
scratch_fallback=
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  scratch_fallback="$attempt/.slurm-tmp-$SLURM_JOB_ID"
  mkdir -m 700 -- "$scratch_fallback"
  export SLURM_TMPDIR="$scratch_fallback"
fi
[[ -d "$SLURM_TMPDIR" && -w "$SLURM_TMPDIR" ]]
# Slurm non-login shells may not inherit the module function.
if ! type module >/dev/null 2>&1; then
  [[ -r /etc/profile.d/z00_lmod.sh ]]
  # System shell initialization uses optional unset variables.
  set +u
  source /etc/profile.d/z00_lmod.sh
  set -u
fi
type module >/dev/null 2>&1
module load python/3.11.15 uv/0.11.2
runtime=$(mktemp -d "$SLURM_TMPDIR/moonwalker-wikiart-026.XXXXXX")
cleanup() { rm -rf -- "$runtime"; if [[ -n "$scratch_fallback" ]]; then rmdir -- "$scratch_fallback"; fi; }
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export UV_CACHE_DIR="$runtime/uv" HF_HOME="$runtime/hf" HF_HUB_CACHE="$runtime/hf/hub"
export TORCH_HOME="$runtime/torch" TORCH_EXTENSIONS_DIR="$runtime/torch-extensions"
export TRITON_CACHE_DIR="$runtime/triton" CUDA_CACHE_PATH="$runtime/cuda"
export WANDB_DIR="$runtime/wandb" WANDB_CACHE_DIR="$runtime/wandb-cache"
export MPLCONFIGDIR="$runtime/mpl" XDG_CACHE_HOME="$runtime/xdg"
export TMPDIR="$runtime/tmp" PIP_CACHE_DIR="$runtime/pip"
mkdir -p "$TMPDIR"
export STEERING_PYTHON="$runtime/venv/bin/python"
uv venv --python "$(command -v python3)" "$runtime/venv"
# CUDA wheels are explicit hash-pinned URLs; all index-resolved packages use PyPI.
sed '\|^--extra-index-url https://download.pytorch.org/whl/cu128$|d' diff_inversion/steering/requirements.txt > "$runtime/requirements-pypi.txt"
uv pip install --python "$STEERING_PYTHON" --default-index https://pypi.org/simple -r "$runtime/requirements-pypi.txt" -r config/steering/runtime_extra_requirements.txt -r config/steering/wikiart_metric_requirements.txt
uv pip freeze --python "$STEERING_PYTHON" > "$attempt/runtime-packages.txt"
module list > "$attempt/runtime-modules.txt" 2>&1
uname -m > "$attempt/runtime-architecture.txt"
"$STEERING_PYTHON" --version > "$attempt/runtime-python.txt"
export PYTHONPATH="$repo"
"$STEERING_PYTHON" scripts/run_wikiart_readers.py "$attempt" "$repo"
