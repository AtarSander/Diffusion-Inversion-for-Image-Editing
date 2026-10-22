#!/usr/bin/env bash
# ABOUTME: Runs the steering pipeline (extract -> fit -> generate) for one concept config.
# ABOUTME: Usage: steering_wikiart_run.sh <config-stem> <gpu>; writes output/steering/<stem>/<ts>/.
set -euo pipefail
concept=$1
gpu=$2
root=/nas/lstanisz/code/lorainv
ts=$(date +%Y%m%d_%H%M%S)
run="$root/output/steering/$concept/$ts"
mkdir -p "$run"
cd "$root"
export CUDA_VISIBLE_DEVICES=$gpu
py="$root/.venv/bin/python"
{
  echo "concept=$concept gpu=$gpu ts=$ts git_sha=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  "$py" -m diff_inversion.steering extract \
    --config "config/steering/$concept.yaml" --output "$run/extract"
  "$py" -m diff_inversion.steering fit \
    --features "$run/extract/features.pt" --output "$run/fit"
  "$py" -m diff_inversion.steering generate \
    --config "config/steering/$concept.yaml" --vectors "$run/fit/directions.pt" \
    --output "$run/generate"
  echo "DONE $run"
} 2>&1 | tee "$run/run.log"
