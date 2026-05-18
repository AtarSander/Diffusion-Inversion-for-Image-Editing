"""Convert per-step latent trajectory files into one stacked tensor per sample.

The old SDXL sample format stores one file per denoising step:

    sample_000000/latents/x_000.pt
    sample_000000/latents/x_001.pt
    ...

The new format stores the same tensors stacked along dimension 0:

    sample_000000/latents/trajectory.pt

For a migrated sample, ``torch.load("trajectory.pt")[i]`` matches the tensor
previously saved in ``x_{i:03d}.pt``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable: Iterable[Path], **_: Any) -> Iterable[Path]:
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate SDXL latent trajectories from x_*.pt files to trajectory.pt."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing sample directories, e.g. .../sdxl_trajectories.",
    )
    parser.add_argument(
        "--sample-glob",
        default="sample_*",
        help="Glob used to find sample directories under root.",
    )
    parser.add_argument(
        "--latents-dir-name",
        default="latents",
        help="Name of the per-sample latent directory.",
    )
    parser.add_argument(
        "--old-glob",
        default="x_*.pt",
        help="Glob for old per-step latent files inside each latent directory.",
    )
    parser.add_argument(
        "--output-name",
        default="trajectory.pt",
        help="Name of the new stacked latent tensor file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild output files that already exist.",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old x_*.pt files after the stacked file is written.",
    )
    parser.add_argument(
        "--verify-exact",
        action="store_true",
        help=(
            "Reload trajectory.pt and compare every step against the old tensors before "
            "deleting. This is very slow on metadata-heavy HPC filesystems."
        ),
    )
    parser.add_argument(
        "--low-inode-mode",
        action="store_true",
        help=(
            "Delete old x_*.pt files after loading them but before writing trajectory.pt. "
            "This can recover from a completely exhausted inode quota, but is less safe "
            "because a failed save would leave that sample only in process memory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without writing or deleting anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most this many samples.",
    )
    parser.add_argument(
        "--no-meta-update",
        action="store_true",
        help="Do not add latents_format/latents_file_name to existing meta.json files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of sample directories to migrate in parallel.",
    )
    return parser.parse_args()


def iter_sample_dirs(root: Path, sample_glob: str) -> Iterable[Path]:
    yield from sorted(path for path in root.glob(sample_glob) if path.is_dir())


def load_old_latents(old_files: list[Path]) -> list[Any]:
    import torch

    return [torch.load(path, map_location="cpu") for path in old_files]


def verify_stacked_file(
    output_path: Path,
    expected_steps: int,
    expected_tensors: list[Any] | None = None,
) -> None:
    import torch

    stacked = torch.load(output_path, map_location="cpu")
    if not isinstance(stacked, torch.Tensor):
        raise TypeError(f"{output_path} did not contain a torch.Tensor")
    if stacked.shape[0] != expected_steps:
        raise ValueError(
            f"{output_path} has {stacked.shape[0]} steps, expected {expected_steps}"
        )
    if expected_tensors is not None:
        for step_idx, expected in enumerate(expected_tensors):
            if not torch.equal(stacked[step_idx], expected):
                raise ValueError(
                    f"{output_path} step {step_idx} does not match the old latent file"
                )


def check_output_written(output_path: Path) -> None:
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    if output_path.stat().st_size == 0:
        raise ValueError(f"{output_path} is empty")


def update_meta(sample_dir: Path, output_name: str, trajectory_length: int) -> None:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["latents_format"] = "stacked_pt"
    meta["latents_file_name"] = output_name
    meta["trajectory_length"] = trajectory_length

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def delete_old_files(old_files: list[Path]) -> None:
    for path in old_files:
        path.unlink()


def migrate_sample(
    sample_dir: Path,
    latents_dir_name: str,
    old_glob: str,
    output_name: str,
    overwrite: bool,
    delete_old: bool,
    low_inode_mode: bool,
    verify_exact: bool,
    dry_run: bool,
    update_meta_file: bool,
) -> str:
    latents_dir = sample_dir / latents_dir_name
    if not latents_dir.is_dir():
        return "missing_latents_dir"

    old_files = sorted(latents_dir.glob(old_glob))
    output_path = latents_dir / output_name

    if not old_files:
        return "no_old_files"

    if output_path.exists() and not overwrite:
        if delete_old:
            if verify_exact:
                tensors = load_old_latents(old_files)
                verify_stacked_file(output_path, len(old_files), tensors)
            else:
                check_output_written(output_path)
            if not dry_run:
                delete_old_files(old_files)
                if update_meta_file:
                    update_meta(sample_dir, output_name, len(old_files))
            return "deleted_old"
        return "already_converted"

    if dry_run:
        return "would_convert"

    import torch

    tensors = load_old_latents(old_files)
    stacked = torch.stack(tensors, dim=0)

    if low_inode_mode:
        if not delete_old:
            raise ValueError("--low-inode-mode requires --delete-old")
        delete_old_files(old_files)
        torch.save(stacked, output_path)
    else:
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        torch.save(stacked, tmp_path)
        tmp_path.replace(output_path)

    if verify_exact:
        verify_stacked_file(output_path, len(tensors), tensors)
    else:
        check_output_written(output_path)

    if delete_old and not low_inode_mode:
        delete_old_files(old_files)

    if update_meta_file:
        update_meta(sample_dir, output_name, len(tensors))

    return "converted"


def migrate_sample_from_args(args: tuple[Any, ...]) -> str:
    return migrate_sample(*args)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if args.low_inode_mode and not args.delete_old:
        raise ValueError("--low-inode-mode requires --delete-old")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    counts: dict[str, int] = {}
    converted_or_planned = 0

    sample_dirs = list(iter_sample_dirs(root, args.sample_glob))
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    migrate_args = [
        (
            sample_dir,
            args.latents_dir_name,
            args.old_glob,
            args.output_name,
            args.overwrite,
            args.delete_old,
            args.low_inode_mode,
            args.verify_exact,
            args.dry_run,
            not args.no_meta_update,
        )
        for sample_dir in sample_dirs
    ]

    if args.workers == 1:
        statuses = (
            migrate_sample_from_args(sample_args)
            for sample_args in tqdm(migrate_args, desc="Migrating samples")
        )
        for status in statuses:
            counts[status] = counts.get(status, 0) + 1
            if status in {"converted", "would_convert", "deleted_old"}:
                converted_or_planned += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            statuses = executor.map(migrate_sample_from_args, migrate_args)
            for status in tqdm(statuses, total=len(migrate_args), desc="Migrating samples"):
                counts[status] = counts.get(status, 0) + 1
                if status in {"converted", "would_convert", "deleted_old"}:
                    converted_or_planned += 1

    print("Migration summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
