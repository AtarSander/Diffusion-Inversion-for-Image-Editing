#!/usr/bin/env bash
# Wait for the external ReCap copy, validate it twice, flatten all/all, then submit r16.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-plgevalfm-gpu-a100}
SLURM_PARTITION=${SLURM_PARTITION:-plgrid-gpu-a100}
POLL_SECONDS=${POLL_SECONDS:-300}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-30441}
DATA_BASE=${DATA_BASE:-/net/tscratch/people/plgatarsander/ZZSN_data/processed/sd15_trajectories_stacked_fp32}
LOG_ROOT=${LOG_ROOT:-/net/tscratch/people/plgatarsander/ZZSN_data/logs/slurm/sd15_recap_copy_watcher}

if [[ ${1:-} != --worker ]]; then
  RUN_LOG_DIR="$LOG_ROOT/$(date +%Y-%m-%d/%H-%M-%S)"
  mkdir -p "$RUN_LOG_DIR"
  exec sbatch --job-name=watch-sd15-recap-copy --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=2-00:00:00 \
    --output="$RUN_LOG_DIR/%j.out" --error="$RUN_LOG_DIR/%j.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",PYTHON="$PYTHON",POLL_SECONDS="$POLL_SECONDS",EXPECTED_SAMPLES="$EXPECTED_SAMPLES",DATA_BASE="$DATA_BASE" \
    "$0" --worker
fi

ROOT="$DATA_BASE/all/all"
required=(final.png prompt.json meta.json timesteps.json latents/trajectory.pt conditioning.pt targets/target_eps.pt)
signature=""

while true; do
  [[ -d "$ROOT" ]] || { echo "Waiting for copied root: $ROOT"; sleep "$POLL_SECONDS"; continue; }
  counts=("$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)")
  for artifact in "${required[@]}"; do
    counts+=("$(find "$ROOT" -type f -path "*/$artifact" | wc -l)")
  done
  current=$(IFS=,; echo "${counts[*]}")
  expected=""
  for _ in "${counts[@]}"; do expected+="${EXPECTED_SAMPLES},"; done
  expected=${expected%,}
  echo "$(date --iso-8601=seconds) counts=$current"
  [[ "$current" == "$expected" && "$signature" == "$current" ]] && break
  signature="$current"
  sleep "$POLL_SECONDS"
done

for sample in "$ROOT"/*; do
  [[ -d "$sample" ]] || continue
  for artifact in "${required[@]}"; do
    [[ -f "$sample/$artifact" ]] || { echo "Incomplete sample after count validation: $sample/$artifact" >&2; exit 1; }
  done
done

outer="$DATA_BASE/all"
tmp="$DATA_BASE/.all-flattening-${SLURM_JOB_ID}"
[[ $(find "$outer" -mindepth 1 -maxdepth 1 -printf '%f\n' | wc -l) -eq 1 ]]
[[ -d "$outer/all" ]]
mv "$outer" "$tmp"
mv "$tmp/all" "$outer"
rmdir "$tmp"
echo "Flattened ReCap root: $outer"
train_submission=$("$REPO_DIR/scripts/launch_train_sd15_diffusiondb_artificial_mixture.sh")
printf '%s\n' "$train_submission"
train_job_id=$(sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' <<<"$train_submission")
[[ -n "$train_job_id" ]] || { echo "Could not parse training job ID." >&2; exit 1; }
"$REPO_DIR/scripts/launch_piebench_sd15_mixture_eval.sh" "$train_job_id"
