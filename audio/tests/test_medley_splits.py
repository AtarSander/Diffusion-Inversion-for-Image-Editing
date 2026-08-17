# ABOUTME: Tests for the benchmark-split resolver and the invariant that makes paired metrics
# ABOUTME: work: a split's reference wavs must carry exactly the names its edits will get.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editing.AudioEditingCode.code.env import (  # noqa: E402
    MEDLEY_SPLIT_CSVS,
    PATH_LOWER_BOUND_MEDLEY,
    PATH_PROMPTS_MEDLEY,
    medley_split_paths,
)


def test_full_split_returns_the_configured_paths_unchanged():
    """Existing runs must keep resolving to exactly what .env says, or old results move."""
    assert medley_split_paths("full") == (PATH_PROMPTS_MEDLEY, PATH_LOWER_BOUND_MEDLEY)


def test_subset_reference_is_a_sibling_of_the_full_one():
    csv, reference = medley_split_paths("hparam")
    assert Path(csv).name == "captions_gpt5_hparam.csv"
    assert Path(csv).parent == Path(PATH_PROMPTS_MEDLEY).parent
    assert Path(reference) == Path(PATH_LOWER_BOUND_MEDLEY).parents[1] / "lower_bound_hparam/audios"


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="Unknown split"):
        medley_split_paths("nope")


@pytest.mark.parametrize("split", sorted(MEDLEY_SPLIT_CSVS))
def test_every_split_csv_keeps_the_full_set_indices(split):
    """Outputs are named a{index}.wav, so a subset's indices must be the full set's, not 0..n-1.

    The `hparam` subset is indexed 0..692 with gaps; naming its reference positionally was what
    made the paired metrics silently refuse to score it.
    """
    frame = pd.read_csv(medley_split_paths(split)[0], index_col=0)
    assert frame.index.is_unique
    assert frame.index.max() <= 695
    if split not in {"full", "tracks"}:
        assert list(frame.index) != list(range(len(frame))), (
            f"{split} looks positionally indexed; the reference naming assumption needs checking"
        )
