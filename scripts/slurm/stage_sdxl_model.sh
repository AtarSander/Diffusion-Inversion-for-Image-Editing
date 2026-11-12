#!/bin/bash

# Stage only the SDXL components used by this project onto node-local storage.
stage_sdxl_model() {
  local model_dir=$1 available_kb required lock_file lock_fd attempt status
  lock_file="${model_dir}.lock"
  mkdir -p "$(dirname "$model_dir")"
  exec {lock_fd}>"$lock_file"
  flock "$lock_fd"
  if [[ -f "$model_dir/model_index.json" && \
        -f "$model_dir/unet/diffusion_pytorch_model.fp16.safetensors" && \
        -f "$model_dir/vae/diffusion_pytorch_model.fp16.safetensors" && \
        -f "$model_dir/text_encoder/model.fp16.safetensors" && \
        -f "$model_dir/text_encoder_2/model.fp16.safetensors" ]]; then
    echo "Reusing staged SDXL at $model_dir"
    flock -u "$lock_fd"
    eval "exec ${lock_fd}>&-"
    return 0
  fi
  rm -rf -- "$model_dir"
  mkdir -p "$model_dir"
  available_kb=$(df -Pk "$model_dir" | awk "NR == 2 {print \$4}")
  (( available_kb >= 12582912 )) || {
    echo "Need at least 12 GiB of node-local scratch; only ${available_kb} KiB available at $model_dir." >&2
    return 1
  }
  echo "Staging SDXL in $model_dir (available local scratch: ${available_kb} KiB)"
  for attempt in 1 2 3; do
    if "$REPO_DIR/.venv/bin/hf" download stabilityai/stable-diffusion-xl-base-1.0 \
      --local-dir "$model_dir" --max-workers 1 \
      --include "model_index.json" \
      --include "scheduler/*" \
      --include "tokenizer/*" \
      --include "tokenizer_2/*" \
      --include "text_encoder/config.json" \
      --include "text_encoder/model.fp16.safetensors" \
      --include "text_encoder_2/config.json" \
      --include "text_encoder_2/model.fp16.safetensors" \
      --include "unet/config.json" \
      --include "unet/diffusion_pytorch_model.fp16.safetensors" \
      --include "vae/config.json" \
      --include "vae/diffusion_pytorch_model.fp16.safetensors"; then
      break
    else
      status=$?
    fi
    echo "SDXL staging attempt $attempt failed with exit status $status." >&2
    (( attempt == 3 )) && return "$status"
    rm -rf -- "$model_dir"
    sleep $((attempt * 30))
  done
  for required in model_index.json unet/diffusion_pytorch_model.fp16.safetensors \
    vae/diffusion_pytorch_model.fp16.safetensors text_encoder/model.fp16.safetensors \
    text_encoder_2/model.fp16.safetensors; do
    [[ -f "$model_dir/$required" ]] || {
      echo "Missing staged SDXL file: $model_dir/$required" >&2
      return 1
    }
  done
  echo "SDXL staging validation complete"
  flock -u "$lock_fd"
  eval "exec ${lock_fd}>&-"
  return 0
}
