#!/bin/bash

# Runtime caches are disposable and belong on the compute node, not shared storage.
# The caller must define JOB_TMP_DIR before sourcing this file.
[[ -n "${JOB_TMP_DIR:-}" ]] || {
  echo "JOB_TMP_DIR must be set before sourcing job_local_cache.sh" >&2
  return 2 2>/dev/null || exit 2
}

JOB_CACHE_ROOT="$JOB_TMP_DIR/cache"
export JOB_CACHE_ROOT
export TMPDIR="$JOB_TMP_DIR/tmp"
export UV_CACHE_DIR="$JOB_CACHE_ROOT/uv"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg"
export HF_HOME="$JOB_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$JOB_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$JOB_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$JOB_CACHE_ROOT/matplotlib"
export WANDB_CACHE_DIR="$JOB_CACHE_ROOT/wandb-cache"
export WANDB_DIR="$JOB_CACHE_ROOT/wandb"
export PIP_CACHE_DIR="$JOB_CACHE_ROOT/pip"
export NUMBA_CACHE_DIR="$JOB_CACHE_ROOT/numba"

mkdir -p \
  "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
  "$HF_HUB_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" \
  "$WANDB_CACHE_DIR" "$WANDB_DIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR"
