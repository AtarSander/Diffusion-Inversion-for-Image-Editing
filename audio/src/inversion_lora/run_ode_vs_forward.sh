#!/bin/bash
# ABOUTME: Run the ODE-vs-forward-diffusion analysis on one node: one shard per visible GPU for
# ABOUTME: each model, then aggregate. PYTHON must be an interpreter with the audio requirements.
#
# Usage (from audio/):  PYTHON=.venv/bin/python bash src/inversion_lora/run_ode_vs_forward.sh
# Optional: MODELS="audioldm2 stable_audio"  NUM_SAMPLES=1000  STEPS=200  OUT_ROOT=output/ode_vs_forward

set -euo pipefail

PYTHON="${PYTHON:?set PYTHON to the audio environment interpreter}"
MODELS="${MODELS:-audioldm2 stable_audio}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
STEPS="${STEPS:-200}"
OUT_ROOT="${OUT_ROOT:-output/ode_vs_forward}"
SCRIPT=src/inversion_lora/analyze_ode_vs_forward.py

mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
fi
NUM_SHARDS="${#GPUS[@]}"
[ "$NUM_SHARDS" -gt 0 ] || { echo "ERROR: no GPUs visible" >&2; exit 1; }
echo "models=[$MODELS] samples=$NUM_SAMPLES steps=$STEPS shards=$NUM_SHARDS gpus=[${GPUS[*]}]"

for model in $MODELS; do
  mkdir -p "$OUT_ROOT/$model"
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    # One process per GPU, pinned by CUDA_VISIBLE_DEVICES so each sees its card as cuda:0.
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON" "$SCRIPT" run --model "$model" \
      --shard "$shard" --num_shards "$NUM_SHARDS" --num_samples "$NUM_SAMPLES" \
      --steps "$STEPS" --device cuda:0 --out_root "$OUT_ROOT" \
      > "$OUT_ROOT/$model/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: a $model shard failed; tails of its logs:" >&2
    tail -n 20 "$OUT_ROOT/$model"/shard_*.log >&2
    exit 1
  fi
  "$PYTHON" "$SCRIPT" aggregate --model "$model" --out_root "$OUT_ROOT"
  cat "$OUT_ROOT/$model/REPORT.md"
done
