# ABOUTME: Pre-submission check for the MedleyMD benchmark: every configured path resolves,
# ABOUTME: all 696 rows point at a real mix, and the model weights are cached for offline nodes.

import os
import sys
from pathlib import Path

import fire
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "editing/AudioEditingCode/code"))

from env import (  # noqa: E402
    ALDM2_TEMP_DIR,
    PATH_AUDIOS_MEDLEY,
    PATH_EDIT_OUTPUTS,
    PATH_LOWER_BOUND_MEDLEY,
    PATH_PROMPTS_MEDLEY,
)

GATED_REPOS = {"stabilityai/stable-audio-open-1.0"}
REPOS = ["cvssp/audioldm2-large", "stabilityai/stable-audio-open-1.0"]


def _repo_cache_dir(repo_id: str) -> Path:
    hf_home = os.environ.get("HF_HOME")
    hub = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache/huggingface/hub"
    return hub / ("models--" + repo_id.replace("/", "--"))


def main(check_lower_bound: bool = False) -> None:
    """Verify everything the baseline sweep needs before submitting to SLURM.

    Args:
        check_lower_bound: Also require the paired reference set (needed for eval, not edits).
    """
    problems: list[str] = []
    print("=== configured paths ===")
    for name, value, must_exist in [
        ("PATH_AUDIOS_MEDLEY", PATH_AUDIOS_MEDLEY, True),
        ("PATH_PROMPTS_MEDLEY", PATH_PROMPTS_MEDLEY, True),
        ("PATH_EDIT_OUTPUTS", PATH_EDIT_OUTPUTS, False),
        ("ALDM2_TEMP_DIR", ALDM2_TEMP_DIR, False),
        ("PATH_LOWER_BOUND_MEDLEY", PATH_LOWER_BOUND_MEDLEY, check_lower_bound),
    ]:
        exists = Path(value).exists()
        print(f"  {'OK ' if exists else '-- '} {name:24s} {value}")
        if must_exist and not exists:
            problems.append(f"{name} does not exist: {value}")

    # Output dirs are created on demand, but a read-only or missing parent fails 72 jobs at once.
    for name, value in [("PATH_EDIT_OUTPUTS", PATH_EDIT_OUTPUTS), ("ALDM2_TEMP_DIR", ALDM2_TEMP_DIR)]:
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
            probe = Path(value) / ".preflight_write_test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            problems.append(f"{name} is not writable: {value} ({exc})")

    print("\n=== benchmark rows ===")
    if Path(PATH_PROMPTS_MEDLEY).exists() and Path(PATH_AUDIOS_MEDLEY).exists():
        df = pd.read_csv(PATH_PROMPTS_MEDLEY, index_col=0, header=0)
        missing = set()
        for _, row in df.iterrows():
            filename = row["filename"]
            path = Path(PATH_AUDIOS_MEDLEY) / filename.split("_MIX")[0] / filename
            if not path.exists():
                missing.add(str(path))
        print(f"  {len(df) - len(missing) if not missing else len(df) - len(missing)}/{len(df)} rows resolve to an existing mix")
        print(f"  {df['edit'].value_counts().to_dict()}")
        if missing:
            problems.append(f"{len(missing)} distinct source files missing, e.g. {sorted(missing)[0]}")
    else:
        print("  skipped (prompts or audio dir missing)")

    print("\n=== model weights cache ===")
    offline = os.environ.get("HF_HUB_OFFLINE")
    print(f"  HF_HOME={os.environ.get('HF_HOME') or '(default)'}  HF_HUB_OFFLINE={offline or '0'}")
    for repo in REPOS:
        cache = _repo_cache_dir(repo)
        cached = cache.exists() and any(cache.rglob("*.safetensors")) or (
            cache.exists() and any(cache.rglob("*.bin"))
        )
        print(f"  {'OK ' if cached else '-- '} {repo}")
        if not cached:
            message = (
                f"{repo} not cached at {cache}; run slurm_scripts/wcss/prefetch_models.py first"
            )
            if offline:
                problems.append(message + " (HF_HUB_OFFLINE is set, so jobs cannot download)")
            else:
                # Nodes with internet will fetch it, but 72 tasks pulling ~12 GB each is slow
                # and rate-limit prone, so this stays a warning rather than a hard stop.
                print(f"      WARNING: {message}")
    if not os.environ.get("HF_TOKEN"):
        problems.append("HF_TOKEN unset; " + ", ".join(GATED_REPOS) + " is license-gated")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S) — do not submit yet:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("PREFLIGHT OK — safe to submit the baseline array")


if __name__ == "__main__":
    fire.Fire(main)
