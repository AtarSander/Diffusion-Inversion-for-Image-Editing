#!/bin/bash

# Stage and validate SD1.5 explicitly in the caller-owned node-local job directory.
stage_sd15_model() {
  local model_dir=$1 available_kb required lock_file lock_fd
  lock_file="${model_dir}.lock"
  mkdir -p "$(dirname "$model_dir")"
  exec {lock_fd}>"$lock_file"
  flock "$lock_fd"
  if [[ -f "$model_dir/model_index.json" && -f "$model_dir/unet/diffusion_pytorch_model.safetensors" && -f "$model_dir/vae/diffusion_pytorch_model.safetensors" && -f "$model_dir/text_encoder/model.safetensors" ]]; then
    echo "Reusing staged SD1.5 at $model_dir"
    flock -u "$lock_fd"
    eval "exec ${lock_fd}>&-"
    return 0
  fi
  rm -rf -- "$model_dir"
  mkdir -p "$model_dir"
  available_kb=$(df -Pk "$model_dir" | awk "NR == 2 {print \$4}")
  (( available_kb >= 5242880 )) || {
    echo "Need at least 5 GiB of node-local scratch; only ${available_kb} KiB available at $model_dir." >&2
    return 1
  }
  echo "Staging SD1.5 in $model_dir (available local scratch: ${available_kb} KiB)"
  local attempt status
  for attempt in 1 2 3; do
    if "$REPO_DIR/.venv/bin/hf" download runwayml/stable-diffusion-v1-5 \
      --local-dir "$model_dir" --max-workers 1 \
      --include "model_index.json" \
      --include "scheduler/*" \
      --include "tokenizer/*" \
      --include "text_encoder/config.json" \
      --include "text_encoder/model.safetensors" \
      --include "unet/config.json" \
      --include "unet/diffusion_pytorch_model.safetensors" \
      --include "vae/config.json" \
      --include "vae/diffusion_pytorch_model.safetensors"; then
      break
    else
      status=$?
    fi
    echo "SD1.5 staging attempt $attempt failed with exit status $status." >&2
    (( attempt == 3 )) && return "$status"
    rm -rf -- "$model_dir"
    sleep $((attempt * 30))
  done
  for required in model_index.json unet/diffusion_pytorch_model.safetensors \
    vae/diffusion_pytorch_model.safetensors text_encoder/model.safetensors; do
    [[ -f "$model_dir/$required" ]] || {
      echo "Missing staged SD1.5 file: $model_dir/$required" >&2
      return 1
    }
  done
  echo "SD1.5 staging validation complete"
  flock -u "$lock_fd"
  eval "exec ${lock_fd}>&-"
  return 0
}
