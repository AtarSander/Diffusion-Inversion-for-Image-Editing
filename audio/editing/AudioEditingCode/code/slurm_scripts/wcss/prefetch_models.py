# ABOUTME: Download every model the MedleyMD baselines need into HF_HOME, on a login node,
# ABOUTME: so compute jobs can run with HF_HUB_OFFLINE=1 instead of hitting the network 72 times.

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# .../code/slurm_scripts/wcss/prefetch_models.py -> .../audio
AUDIO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(AUDIO_ROOT / ".env", override=False)

# Stable Audio Open is license-gated: accept the terms on the model page first, then the token
# authorises the download. AudioLDM2 is open.
REPOS = [
    ("cvssp/audioldm2-large", False),
    ("stabilityai/stable-audio-open-1.0", True),
]


def main() -> int:
    """Fetch each repo into the configured HF cache, reporting what is missing."""
    hf_home = os.environ.get("HF_HOME")
    token = os.environ.get("HF_TOKEN")
    print(f"HF_HOME  : {hf_home or '(default ~/.cache/huggingface)'}")
    print(f"HF_TOKEN : {'set' if token else 'NOT SET'}")
    if os.environ.get("HF_HUB_OFFLINE"):
        print("HF_HUB_OFFLINE is set; unset it on the login node to download.", file=sys.stderr)
        return 2

    failures = []
    for repo_id, gated in REPOS:
        print(f"\n=== {repo_id}{' (gated)' if gated else ''}")
        if gated and not token:
            failures.append(f"{repo_id}: HF_TOKEN required")
            print("  SKIP: no HF_TOKEN", file=sys.stderr)
            continue
        try:
            path = snapshot_download(repo_id=repo_id, token=token if gated else None)
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            print(f"  OK {path}  ({size / 1e9:.1f} GB)")
        except Exception as exc:  # noqa: BLE001 - report every repo, then fail loudly below
            failures.append(f"{repo_id}: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} repo(s) unavailable:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll models cached. Compute jobs can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
