# ABOUTME: Tests for the consolidated metrics layout: the migration must be lossless, must keep
# ABOUTME: the legacy files when it is not, and must not confuse the two per-example key schemes.

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editing.run_metrics import (  # noqa: E402
    AGGREGATE_JSON,
    PER_EXAMPLE_CSV,
    aggregates,
    migrate,
    per_example,
)

ROWS = [178, 469, 3]


def make_legacy(tmp_path: Path) -> Path:
    """A run in the old eight-file layout, with both key conventions represented."""
    run = tmp_path / "run"
    run.mkdir()
    n = len(ROWS)
    pd.DataFrame({"audio_idx": range(n), "lpaps": [4.1, 5.2, 6.3],
                  "classification_task": ["GENRE", "INSTR", "GENRE"]}).to_csv(
        run / "lpaps_to_source.csv")
    pd.DataFrame({"clap": [0.11, 0.22, 0.33], "prompt": ["a", "b", "c"]},
                 index=range(n)).to_csv(run / "clap_to_target_prompt.csv")
    pd.DataFrame({"muqt_sim_p0": [0.4, 0.5, 0.6], "p0": ["a", "b", "c"]},
                 index=range(n)).to_csv(run / "mulan_to_target_prompt.csv")
    pd.DataFrame({"clap_dir": [0.01, 0.02, 0.03], "mulan_dir": [0.04, 0.05, 0.06]},
                 index=range(n)).to_csv(run / "directional_to_prompts.csv")
    # PSNR/SSIM is keyed by wav name, i.e. the row's index in the split, not by position.
    pd.DataFrame({"psnr": [19.1, 18.2, 17.3], "ssim": [0.61, 0.62, 0.63]},
                 index=[f"a{i}.wav" for i in ROWS]).to_csv(run / "psnr_ssim_per_file.csv",
                                                           index_label="filename")
    (run / "final_results.json").write_text(json.dumps({"LPAPS": {"mean": 5.2}}))
    (run / "per_task_results.json").write_text(json.dumps({"GENRE": {"LPAPS": {"mean": 5.2}}}))
    (run / "source_distance_metrics.json").write_text(json.dumps({"psnr": "18.2"}))
    return run


def test_migration_preserves_every_value(tmp_path):
    run = make_legacy(tmp_path)
    migrate(str(run), remove=True)

    assert per_example(run, "lpaps").tolist() == [4.1, 5.2, 6.3]
    assert per_example(run, "clap").tolist() == [0.11, 0.22, 0.33]
    assert per_example(run, "muqt_sim_p0").tolist() == [0.4, 0.5, 0.6]
    assert per_example(run, "mulan_dir").tolist() == [0.04, 0.05, 0.06]
    assert aggregates(run)["source_distance"] == {"psnr": "18.2"}
    assert aggregates(run)["final"]["LPAPS"]["mean"] == 5.2


def test_migration_leaves_psnr_alone_and_keeps_its_own_key(tmp_path):
    """The two tables use different keys, so the reader must report which one it returned."""
    run = make_legacy(tmp_path)
    migrate(str(run), remove=True)

    assert (run / "psnr_ssim_per_file.csv").exists()
    psnr = per_example(run, "psnr")
    assert list(psnr.index) == sorted(ROWS)
    assert psnr.index.name == "row_idx"
    assert per_example(run, "lpaps").index.name == "position"


def test_migration_removes_exactly_the_seven_legacy_files(tmp_path):
    run = make_legacy(tmp_path)
    assert len(list(run.iterdir())) == 8
    migrate(str(run), remove=True)
    assert sorted(p.name for p in run.iterdir()) == [
        AGGREGATE_JSON, PER_EXAMPLE_CSV, "psnr_ssim_per_file.csv"
    ]


def test_migration_keeps_the_originals_when_a_value_does_not_round_trip(tmp_path, monkeypatch):
    """The delete is irreversible, so a failed check must leave the only copy in place."""
    run = make_legacy(tmp_path)
    real_to_csv = pd.DataFrame.to_csv

    def corrupt(self, *args, **kwargs):
        if args and str(args[0]).endswith(PER_EXAMPLE_CSV):
            self = self.copy()
            self["lpaps"] = self["lpaps"] + 1.0
        return real_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", corrupt)
    with pytest.raises(RuntimeError, match="did not round-trip"):
        migrate(str(run), remove=True)
    assert (run / "lpaps_to_source.csv").exists()


def test_reader_names_the_migration_when_a_run_is_still_legacy(tmp_path):
    run = make_legacy(tmp_path)
    with pytest.raises(FileNotFoundError, match="migrate_all"):
        per_example(run, "lpaps")
