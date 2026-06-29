#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export SPLIT=${SPLIT:-val}
export CHUNK_SIZE=${CHUNK_SIZE:-256}
export ARRAY_PARALLELISM=${ARRAY_PARALLELISM:-8}
export JOB_NAME=${JOB_NAME:-sd15_sample_gather_val}

exec "$SCRIPT_DIR/launch_sd15_sample_gather_submitit.sh" "$@"
