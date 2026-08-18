# ABOUTME: Pack a run's per-example wavs into one tar so a finished run costs a handful of inodes
# ABOUTME: instead of hundreds, and unpack it to node-local scratch for the metrics that need files.

import shutil
import tarfile
from pathlib import Path

import fire

ARCHIVE_NAME = "audios.tar"


def _wavs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.wav") if p.is_file())


def pack(run_dir: str, remove: bool = False, overwrite: bool = False) -> Path:
    """Tar `<run_dir>/audios` into `<run_dir>/audios.tar`.

    Uncompressed: the payload is PCM that gzip barely shrinks, and an uncompressed member can be
    extracted individually without reading the whole archive.

    Args:
        run_dir: Run directory holding `audios/`.
        remove: Delete `audios/` once the archive is verified to hold every file, byte for byte.
        overwrite: Replace an existing archive instead of refusing.

    Returns:
        Path of the archive.
    """
    run_path = Path(run_dir)
    source = run_path / "audios"
    archive = run_path / ARCHIVE_NAME
    if not source.is_dir():
        raise FileNotFoundError(f"No audios/ directory in {run_path}")
    if archive.exists() and not overwrite:
        raise FileExistsError(f"{archive} exists; pass --overwrite to replace it")

    files = _wavs(source)
    if not files:
        raise FileNotFoundError(f"No wavs to pack in {source}")

    staging = run_path / f".{ARCHIVE_NAME}.partial"
    with tarfile.open(staging, "w") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
    staging.replace(archive)

    # Verify against the archive rather than trusting the write: `remove` deletes the only copy.
    with tarfile.open(archive) as tar:
        members = {m.name: m.size for m in tar.getmembers() if m.isfile()}
    expected = {p.name: p.stat().st_size for p in files}
    if members != expected:
        missing = sorted(set(expected) - set(members))
        raise RuntimeError(
            f"{archive} does not match {source}: {len(members)} members vs {len(expected)} files"
            + (f", missing e.g. {missing[:3]}" if missing else ", sizes differ")
        )

    print(f"packed {len(files)} wavs -> {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    if remove:
        shutil.rmtree(source)
        print(f"removed {source} ({len(files)} inodes freed)")
    return archive


def unpack(run_dir: str, dest: str) -> Path:
    """Extract `<run_dir>/audios.tar` into `<dest>/audios`.

    The metrics walk a directory of wavs -- audioldm_eval's PSNR/SSIM and FAD passes are vendored
    and take directory paths -- so scoring an archived run means materialising it somewhere. Point
    `dest` at node-local scratch and the 32 kHz resample cache, which `ensure_resampled` writes as
    a sibling of the audio directory, lands there too instead of on the shared filesystem.

    Args:
        run_dir: Run directory holding `audios.tar`.
        dest: Directory to extract into; `<dest>/audios` is created.

    Returns:
        Path of the extracted audio directory.
    """
    archive = Path(run_dir) / ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(f"No {ARCHIVE_NAME} in {run_dir}")
    target = Path(dest) / "audios"
    target.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in members:
            if Path(member.name).name != member.name:
                raise ValueError(f"{archive} holds a path, not a bare name: {member.name!r}")
        tar.extractall(target, members=members)

    found = len(_wavs(target))
    if found != len(members):
        raise RuntimeError(f"extracted {found} wavs from {archive}, expected {len(members)}")
    print(f"unpacked {found} wavs -> {target}")
    return target


if __name__ == "__main__":
    fire.Fire({"pack": pack, "unpack": unpack})
