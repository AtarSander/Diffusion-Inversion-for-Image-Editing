# ABOUTME: Tests for the resume guard used by --skip_existing, and for why it relies on outputs
# ABOUTME: being written atomically rather than on validating them after the fact.

import sys
from pathlib import Path

import pytest
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "editing/AudioEditingCode/code"))

from edit_audioldm_medleydb import is_complete_wav  # noqa: E402


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "a0.wav"
    torchaudio.save(str(path), torch.rand(1, 16000) * 2 - 1, sample_rate=16000)
    return path


def test_complete_wav_is_skippable(wav):
    assert is_complete_wav(wav)


def test_missing_file_is_not_skippable(tmp_path):
    assert not is_complete_wav(tmp_path / "absent.wav")


def test_truncated_wav_cannot_be_detected_from_its_contents(wav):
    """Why outputs are written atomically instead of validated afterwards.

    torchaudio reports the frames actually present, so a half-written file is indistinguishable
    from a legitimately shorter one. The rename is what guarantees a present file is complete.
    """
    data = wav.read_bytes()
    wav.write_bytes(data[: len(data) // 2])
    assert torchaudio.info(str(wav)).num_frames < 16000
    assert is_complete_wav(wav), "cannot be caught here; the atomic write is what prevents it"


def test_empty_file_is_not_skippable(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    assert not is_complete_wav(path)


def test_header_only_file_is_not_skippable(wav):
    wav.write_bytes(wav.read_bytes()[:44])
    assert not is_complete_wav(wav)
