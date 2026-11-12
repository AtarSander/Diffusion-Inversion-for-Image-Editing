#!/usr/bin/env bash
# Source after inspecting the site's mkdata/mkvenv conventions and selecting storage.
# Require explicit existing helper-selected data and Lustre quota mount destinations.
: "${DCI_PROJECT_DATA:?Set DCI_PROJECT_DATA to the resolved helper-selected data root}"
: "${DCI_QUOTA_MOUNT:?Set DCI_QUOTA_MOUNT to the Lustre mount used for per-user quota}"
DCI_PROJECT_DATA=$(realpath -e -- "$DCI_PROJECT_DATA") || return 1
case "$DCI_PROJECT_DATA" in
  "$HOME"|"$HOME"/*) echo 'DCI payloads must be outside home.' >&2; return 1 ;;
esac
DCI_ACQUISITION=${DCI_ACQUISITION:-$DCI_PROJECT_DATA/interim/dci_acquisition}
DCI_ACQUISITION=$(realpath -m -- "$DCI_ACQUISITION") || return 1
case "$DCI_ACQUISITION" in
  "$DCI_PROJECT_DATA"/*) ;;
  *) echo 'DCI_ACQUISITION must resolve beneath DCI_PROJECT_DATA.' >&2; return 1 ;;
esac
export DCI_PROJECT_DATA DCI_ACQUISITION DCI_QUOTA_MOUNT
export TMPDIR="$DCI_ACQUISITION/tmp" TMP="$DCI_ACQUISITION/tmp" TEMP="$DCI_ACQUISITION/tmp"
export XDG_CACHE_HOME="$DCI_ACQUISITION/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch" UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export WANDB_DIR="$DCI_ACQUISITION/logs/wandb" WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$DCI_ACQUISITION/logs"
