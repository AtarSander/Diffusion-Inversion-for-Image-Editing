#!/usr/bin/env bash
# ABOUTME: Chained outline-triangles steering run: extract -> fit -> edit circles ->
# ABOUTME: generate -> size scores -> SV comparison vs the filled-triangle directions.
set -euo pipefail
cd /nas/lstanisz/code/lorainv
TS=$(date +%Y%m%d_%H%M%S)
BASE=output/steering/triangles_outline/$TS
EDIT=output/steering/circles_outline_edit/$TS
mkdir -p "$BASE"
export CUDA_VISIBLE_DEVICES=${GPU:-4}
FILLED_FIT=output/steering/triangles_size/20261026_172253/fit/directions.pt
{
  echo "git_sha: $(git rev-parse HEAD)  gpu: $CUDA_VISIBLE_DEVICES  ts: $TS"
  uv run python -m diff_inversion.steering extract \
    --config config/steering/triangles_outline.yaml --output "$BASE/extract"
  uv run python -m diff_inversion.steering fit \
    --features "$BASE/extract/features.pt" --output "$BASE/fit"
  uv run scripts/compare_sv_directions.py --a "$FILLED_FIT" \
    --b "$BASE/fit/directions.pt" --name_a filled --name_b outline
  uv run python -m diff_inversion.steering edit \
    --config config/steering/circles_size_edit.yaml \
    --vectors "$BASE/fit/directions.pt" --output "$EDIT/edit"
  uv run scripts/score_shape_size.py --run_dir "$EDIT/edit"
  uv run python -m diff_inversion.steering generate \
    --config config/steering/triangles_outline.yaml \
    --vectors "$BASE/fit/directions.pt" --output "$BASE/generate"
  echo "ALL STAGES DONE"
} 2>&1 | tee "$BASE/run.log"
