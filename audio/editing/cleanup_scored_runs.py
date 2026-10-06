# ABOUTME: Frees disk by deleting the audio of edit runs that are fully scored, keeping every metric
# ABOUTME: file. Dry run by default; --apply deletes and writes a manifest of what went.

import csv
import json
import shutil
import time
from pathlib import Path

import fire

# What the MedleyMD eval writes for a complete run (eval_medley.main + run_metrics).
FINAL_METRICS = ("LPAPS", "CLAP", "MUQT", "CLAP_DIR", "MUQT_DIR")
SOURCE_METRICS = ("psnr", "ssim")


def loose_wavs(run: Path) -> int | None:
    """Number of wavs in `audios/`, or None when the run keeps only `audios.tar`.

    The archive is deliberately not opened: listing a tar's members on Lustre reads far more than
    its headers (a dry run over the edits root pulled 400 GB before it was stopped).
    """
    if not (run / "audios").is_dir():
        return None
    return sum(1 for _ in (run / "audios").glob("*.wav"))


def csv_rows(path: Path) -> int:
    """Data rows in a CSV, or -1 when it is missing."""
    if not path.is_file():
        return -1
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def unscored_reason(run: Path) -> str | None:
    """Why a run is not fully scored, or None when every metric covers every edit.

    The MedleyMD eval loads `a{idx}.wav` for every row of its split (a missing edit crashes it),
    asserts the per-example table has one row per split row, and that psnr_ssim_per_file has one
    row per edited wav. So a complete metrics.json plus two per-example tables of equal length
    means every edit was scored; loose wavs, when present, must match that count too.
    """
    metrics_path = run / "metrics.json"
    if not metrics_path.is_file():
        return "no metrics.json"
    metrics = json.loads(metrics_path.read_text())
    missing = [m for m in FINAL_METRICS if m not in metrics.get("final", {})]
    missing += [m for m in SOURCE_METRICS if m not in metrics.get("source_distance", {})]
    if missing:
        return f"metrics.json lacks {missing}"
    if any(str(metrics["source_distance"][m]).startswith("-1") for m in SOURCE_METRICS):
        return "psnr/ssim hold the -1 failure sentinel"
    rows = csv_rows(run / "per_example_metrics.csv")
    if rows <= 0 or csv_rows(run / "psnr_ssim_per_file.csv") != rows:
        return "per-example tables missing or of unequal length"
    wavs = loose_wavs(run)
    if wavs is not None and wavs != rows:
        return "loose wavs differ from the scored rows"
    return None


def audio_paths(run: Path) -> list[Path]:
    """The deletable audio of a run: the archive, the wav directory and resample caches."""
    paths = [run / "audios.tar", run / "audios"] + sorted(run.glob("audios_*k"))
    return [p for p in paths if p.exists() and not p.is_symlink()]


def size_of(path: Path) -> int:
    """Apparent size in bytes of a file or directory tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def main(root: str = "outputs/edits", apply: bool = False, keep_list: str | None = None) -> None:
    """List (or delete) the audio of every fully scored run under `root`.

    A run is any directory holding `audios.tar` or `audios/`. Its audio is deleted only when it
    is fully scored (`unscored_reason`). Metric files are never touched.

    Args:
        root: Edits root, relative to audio/ or absolute.
        apply: Delete for real; otherwise only report.
        keep_list: Optional file of run names (one per line) to keep regardless.
    """
    root_path = Path(root).resolve()
    keep = set()
    if keep_list:
        lines = Path(keep_list).read_text().splitlines()
        keep = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    runs = sorted(
        {p.parent for p in root_path.rglob("audios.tar")}
        | {p.parent for p in root_path.rglob("audios") if p.is_dir()}
    )
    candidates, skipped = [], {}
    for run in runs:
        if run.name in keep:
            skipped.setdefault("in keep list", []).append(run)
            continue
        reason = unscored_reason(run)
        if reason:
            skipped.setdefault(reason.split(" lacks ")[0], []).append(run)
            continue
        paths = audio_paths(run)
        candidates.append((run, paths, sum(size_of(p) for p in paths)))

    total = sum(size for _, _, size in candidates)
    print(f"{len(runs)} runs under {root_path}")
    print(f"fully scored, audio to delete: {len(candidates)} runs, {total / 1e9:.1f} GB")
    for reason, items in sorted(skipped.items()):
        print(f"kept ({reason}): {len(items)} runs, e.g. {items[0].relative_to(root_path)}")
    by_parent = {}
    for run, _, size in candidates:
        key = str(run.parent.relative_to(root_path))
        by_parent[key] = by_parent.get(key, 0) + size
    for key, size in sorted(by_parent.items(), key=lambda kv: -kv[1]):
        print(f"  {key:45s} {size / 1e9:8.1f} GB")

    if not apply:
        print("dry run: nothing deleted (pass --apply to delete)")
        return
    manifest = {"root": str(root_path), "deleted": [], "deleted_at": time.strftime("%F %T")}
    for run, paths, size in candidates:
        for path in paths:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        manifest["deleted"].append(
            {
                "run": str(run.relative_to(root_path)),
                "paths": [p.name for p in paths],
                "bytes": size,
            }
        )
    out = root_path / f"cleanup_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"deleted {total / 1e9:.1f} GB from {len(candidates)} runs; manifest {out}")


if __name__ == "__main__":
    fire.Fire(main)
