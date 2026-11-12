#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=$ROOT/external/ovg
REV=e13fe392ba8f0738ce2f041ee356bb2b5a0171db
if [[ ! -d "$DEST" ]]; then
  git clone https://github.com/nicoboou/ovg.git "$DEST"
  git -C "$DEST" checkout --detach "$REV"
fi
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$REV" ]] || {
  echo "OVG checkout must be at $REV; refusing to overwrite an existing checkout." >&2
  exit 1
}
[[ -z "$(git -C "$DEST" status --porcelain)" ]] || {
  echo "OVG checkout has local changes; refusing to use modified upstream code." >&2
  exit 1
}
echo "OVG ready: $DEST ($REV)"
