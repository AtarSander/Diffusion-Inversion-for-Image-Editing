# ABOUTME: Tests for the generated-input split builder: grouping by (track, source caption),
# ABOUTME: row-index preservation, pinned seeds and real-track durations.

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchaudio

AUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.inversion_lora.generate_eval_inputs import build_gen_csv  # noqa: E402


@pytest.fixture()
def real_root(tmp_path: Path) -> Path:
    root = tmp_path / "mixes"
    for track, seconds in [("TrackA", 0.5), ("TrackB", 1.0)]:
        (root / track).mkdir(parents=True)
        wav = torch.zeros(2, int(44100 * seconds))
        torchaudio.save(str(root / track / f"{track}_MIX.wav"), wav, sample_rate=44100)
    return root


def test_build_gen_csv_groups_and_preserves_indices(tmp_path: Path, real_root: Path) -> None:
    src = pd.DataFrame(
        {
            "filename": ["TrackA_MIX.wav"] * 3 + ["TrackB_MIX.wav"],
            "source_captions": ["cap one", "cap one", "cap two", "cap three"],
            "target_captions": ["t0", "t1", "t2", "t3"],
            "edit": ["GENRE", "INSTR", "GENRE", "MOOD"],
        },
        index=[7, 21, 40, 3],
    )
    src_csv = tmp_path / "split.csv"
    src.to_csv(src_csv)

    out = build_gen_csv(src_csv, real_root, tmp_path / "gen.csv", seed_base=1000)

    # Rows sharing (track, source caption) share one clip; a new caption gets a new one.
    assert list(out.index) == [7, 21, 40, 3]
    assert list(out["filename"]) == [
        "gen000_MIX.wav",
        "gen000_MIX.wav",
        "gen001_MIX.wav",
        "gen002_MIX.wav",
    ]
    assert list(out["seed"]) == [1000, 1000, 1001, 1002]
    assert list(out["duration_s"]) == [0.5, 0.5, 0.5, 1.0]
    assert (out["target_captions"] == src["target_captions"]).all()
    assert (out["edit"] == src["edit"]).all()
    assert list(out["real_filename"]) == list(src["filename"])

    # Written CSV round-trips with the same index column.
    reread = pd.read_csv(tmp_path / "gen.csv", index_col=0)
    assert list(reread.index) == [7, 21, 40, 3]
