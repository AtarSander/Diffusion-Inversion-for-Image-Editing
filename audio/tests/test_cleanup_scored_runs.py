# ABOUTME: Checks that cleanup_scored_runs deletes audio only from fully scored runs and never
# ABOUTME: touches metric files or runs with a missing metric or a short per-example table.

import json
import sys
import tarfile
from pathlib import Path

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))

from editing.cleanup_scored_runs import main  # noqa: E402

WAVS = 3


def make_run(
    root: Path, name: str, *, tar: bool = True, drop_metric: bool = False, rows: int = WAVS
):
    """A run directory shaped like the MedleyMD eval output."""
    run = root / "medleymd" / "model" / name
    audios = run / "audios"
    audios.mkdir(parents=True)
    for i in range(WAVS):
        (audios / f"a{i}.wav").write_bytes(b"RIFF" + bytes(100))
    if tar:
        with tarfile.open(run / "audios.tar", "w") as archive:
            for wav in sorted(audios.glob("*.wav")):
                archive.add(wav, arcname=wav.name)
        for wav in audios.glob("*.wav"):
            wav.unlink()
        audios.rmdir()
    final = {m: {"mean": 1.0} for m in ("LPAPS", "CLAP", "MUQT", "CLAP_DIR", "MUQT_DIR")}
    if drop_metric:
        final.pop("MUQT")
    metrics = {"final": final, "source_distance": {"psnr": "20.0", "ssim": "0.5"}}
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "per_example_metrics.csv").write_text("h\n" + "x\n" * rows)
    (run / "psnr_ssim_per_file.csv").write_text("h\n" + "x\n" * WAVS)
    return run


def test_only_fully_scored_runs_lose_their_audio(tmp_path):
    scored_tar = make_run(tmp_path, "scored_tar")
    scored_dir = make_run(tmp_path, "scored_dir", tar=False)
    missing = make_run(tmp_path, "missing_metric", drop_metric=True)
    short = make_run(tmp_path, "short_table", rows=WAVS - 1)
    kept = make_run(tmp_path, "kept_by_list")
    keep_list = tmp_path / "keep.txt"
    keep_list.write_text("kept_by_list\n")

    main(root=str(tmp_path), keep_list=str(keep_list))  # dry run deletes nothing
    assert (scored_tar / "audios.tar").exists() and (scored_dir / "audios").exists()

    main(root=str(tmp_path), apply=True, keep_list=str(keep_list))
    assert not (scored_tar / "audios.tar").exists()
    assert not (scored_dir / "audios").exists()
    for run in (missing, short, kept):
        assert (run / "audios.tar").exists(), run.name
    for run in (scored_tar, scored_dir, missing, short, kept):
        assert (run / "metrics.json").exists() and (run / "per_example_metrics.csv").exists()
    assert len(list(tmp_path.glob("cleanup_manifest_*.json"))) == 1
