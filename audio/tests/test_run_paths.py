# ABOUTME: Tests for the edit-output path builder, pinning the failure that once wrote 696 files
# ABOUTME: into the parent directory because an empty run name silently collapsed the path.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "editing/AudioEditingCode/code"))

from run_paths import resolve_run_dir  # noqa: E402

ROOT = "/tmp/edits"


def test_normal_run_directory():
    out = resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", "myrun")
    assert out == Path("/tmp/edits/medleymd/audioldm2_ddim/myrun/audios")


def test_empty_run_name_is_rejected():
    """The real incident: --run_name "" collapsed the path and 12 shards shared one directory."""
    with pytest.raises(ValueError, match="Empty path component"):
        resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", "")


def test_whitespace_run_name_is_rejected():
    with pytest.raises(ValueError, match="Empty path component"):
        resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", "   ")


def test_none_component_is_rejected():
    with pytest.raises(ValueError, match="Empty path component"):
        resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", None)


@pytest.mark.parametrize("escaping", ["../elsewhere", "/absolute/run", "a/../../.."])
def test_components_cannot_escape_the_outputs_root(escaping):
    with pytest.raises(ValueError, match="escape|not inside"):
        resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", escaping)


def test_result_is_always_under_the_root():
    out = resolve_run_dir(ROOT, "medleymd", "stable_audio", "run")
    assert Path(ROOT).resolve() in out.parents


def test_nested_run_name_is_allowed():
    """Slashes inside a run name are fine as long as they stay inside the root."""
    out = resolve_run_dir(ROOT, "medleymd", "audioldm2_ddim", "sweep/run1")
    assert out == Path("/tmp/edits/medleymd/audioldm2_ddim/sweep/run1/audios")
