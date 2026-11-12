#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
[[ -x $PYTHON ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
cd "$REPO_DIR"
exec "$PYTHON" -m diff_inversion.data.prepare_diffusiondb_artificial_prompts "$@"
