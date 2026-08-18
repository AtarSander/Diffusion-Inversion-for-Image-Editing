# ABOUTME: Tests for the run archiver, above all that it never deletes the audio directory unless
# ABOUTME: the archive provably holds every file at the right size.

import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editing.archive_run import pack, unpack  # noqa: E402


def make_run(tmp_path: Path, count: int = 5) -> Path:
    audios = tmp_path / "run" / "audios"
    audios.mkdir(parents=True)
    for idx in (178, 469, 3, 41, 692)[:count]:
        (audios / f"a{idx}.wav").write_bytes(b"RIFF" + bytes(idx % 251) )
    return tmp_path / "run"


def test_pack_then_unpack_round_trips(tmp_path):
    run = make_run(tmp_path)
    original = {p.name: p.read_bytes() for p in (run / "audios").glob("*.wav")}
    pack(str(run), remove=True)
    assert not (run / "audios").exists()

    out = unpack(str(run), str(tmp_path / "scratch"))
    assert {p.name: p.read_bytes() for p in out.glob("*.wav")} == original


def test_pack_refuses_to_overwrite_silently(tmp_path):
    run = make_run(tmp_path)
    pack(str(run))
    with pytest.raises(FileExistsError):
        pack(str(run))
    pack(str(run), overwrite=True)


def test_pack_keeps_the_directory_when_the_archive_is_short(tmp_path, monkeypatch):
    """The delete is the dangerous step: a truncated archive must not cost the only copy."""
    run = make_run(tmp_path)
    real_add = tarfile.TarFile.add

    def drop_one(self, name, arcname=None, **kwargs):
        if Path(name).name == "a469.wav":
            return
        return real_add(self, name, arcname=arcname, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", drop_one)
    with pytest.raises(RuntimeError, match="does not match"):
        pack(str(run), remove=True)
    assert (run / "audios").exists()
    assert len(list((run / "audios").glob("*.wav"))) == 5


def test_unpack_rejects_paths_in_member_names(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "evil.wav").write_bytes(b"x")
    with tarfile.open(run / "audios.tar", "w") as tar:
        tar.add(tmp_path / "evil.wav", arcname="../escaped.wav")
    with pytest.raises(ValueError, match="path, not a bare name"):
        unpack(str(run), str(tmp_path / "scratch"))


def test_pack_fails_loudly_on_an_empty_run(tmp_path):
    run = tmp_path / "run"
    (run / "audios").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        pack(str(run))
