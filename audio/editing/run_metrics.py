# ABOUTME: One per-example table and one aggregate file per scored run, replacing the eight files
# ABOUTME: the eval used to scatter, with the reader helpers and a migration for scored runs.

import json
from pathlib import Path

import fire
import pandas as pd

PER_EXAMPLE_CSV = "per_example_metrics.csv"
AGGREGATE_JSON = "metrics.json"
# PSNR/SSIM keeps its own file: it is keyed by wav filename, i.e. the row's index in the split,
# while the prompt metrics are keyed by position in the split. Joining the two needs the split
# CSV to map one to the other, which is not worth one inode per run.
PSNR_CSV = "psnr_ssim_per_file.csv"

PROMPT_METRICS = ["lpaps", "clap", "muqt_sim_p0", "clap_dir", "mulan_dir"]
LEGACY_PER_EXAMPLE = {
    "lpaps_to_source.csv": ["lpaps", "classification_task"],
    "clap_to_target_prompt.csv": ["clap"],
    "mulan_to_target_prompt.csv": ["muqt_sim_p0"],
    "directional_to_prompts.csv": ["clap_dir", "mulan_dir"],
}
LEGACY_AGGREGATE = {
    "final_results.json": "final",
    "per_task_results.json": "per_task",
    "source_distance_metrics.json": "source_distance",
}


def write_per_example(run_dir: str | Path, frame: pd.DataFrame) -> Path:
    """Write the consolidated per-example table, keyed by position in the split."""
    path = Path(run_dir) / PER_EXAMPLE_CSV
    frame.to_csv(path, index_label="position")
    return path


def write_aggregates(run_dir: str | Path, aggregates: dict) -> Path:
    """Write the aggregate metrics as one JSON."""
    path = Path(run_dir) / AGGREGATE_JSON
    with path.open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2)
    return path


def per_example(run_dir: str | Path, metric: str) -> pd.Series:
    """One metric's per-example values, indexed so that runs can be paired row by row.

    The index name says which key it is: `position` for the prompt metrics, `row_idx` for
    PSNR/SSIM. Pair like with like -- comparing two runs' `lpaps`, or two runs' `psnr` -- and
    never join one against the other, which would align position 3 with row 3 of the full set.

    Args:
        run_dir: Directory holding the run's metrics.
        metric: Column name, e.g. `lpaps`, `clap`, `psnr`.

    Returns:
        The column, ascending by its key.
    """
    run_path = Path(run_dir)
    if metric in ("psnr", "ssim"):
        frame = pd.read_csv(run_path / PSNR_CSV, index_col=0)
        frame.index = [int(str(i).removeprefix("a").removesuffix(".wav")) for i in frame.index]
        frame.index.name = "row_idx"
        return frame[metric].sort_index()

    path = run_path / PER_EXAMPLE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Runs scored before the metrics were consolidated need "
            f"`python -m editing.run_metrics migrate_all --root <edits root> --remove True`."
        )
    frame = pd.read_csv(path, index_col=0)
    frame.index.name = "position"
    if metric not in frame.columns:
        raise KeyError(f"{path} has no column {metric!r}; available: {list(frame.columns)}")
    return frame[metric].sort_index()


def aggregates(run_dir: str | Path) -> dict:
    """The run's aggregate metrics: `final`, `per_task` and `source_distance`."""
    return json.loads((Path(run_dir) / AGGREGATE_JSON).read_text())


def migrate(run_dir: str, remove: bool = False) -> Path:
    """Fold a run's seven legacy metric files into the two consolidated ones.

    `psnr_ssim_per_file.csv` is left alone; everything else is merged and, once verified, removed.

    Args:
        run_dir: Directory holding the legacy files.
        remove: Delete the legacy files once every value round-trips.

    Returns:
        Path of the consolidated per-example table.
    """
    run_path = Path(run_dir)
    columns = {}
    for filename, wanted in LEGACY_PER_EXAMPLE.items():
        path = run_path / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0).sort_index()
        for column in wanted:
            if column in frame.columns:
                columns[column] = frame[column]
    if not columns:
        raise FileNotFoundError(f"No legacy per-example CSVs in {run_path}")

    combined = pd.DataFrame(columns)
    payload = {
        key: json.loads((run_path / filename).read_text())
        for filename, key in LEGACY_AGGREGATE.items()
        if (run_path / filename).exists()
    }
    write_per_example(run_path, combined)
    write_aggregates(run_path, payload)

    # Verify before deleting: this is the only copy of numbers that cost GPU-hours.
    check = pd.read_csv(run_path / PER_EXAMPLE_CSV, index_col=0)
    for column, series in columns.items():
        if column == "classification_task":
            assert list(check[column]) == list(series), f"{run_path}: {column} changed"
        elif not check[column].sub(series).abs().max() < 1e-12:
            raise RuntimeError(f"{run_path}: {column} did not round-trip; keeping originals")
    if json.loads((run_path / AGGREGATE_JSON).read_text()) != payload:
        raise RuntimeError(f"{run_path}: aggregates did not round-trip; keeping originals")

    if remove:
        for filename in list(LEGACY_PER_EXAMPLE) + list(LEGACY_AGGREGATE):
            (run_path / filename).unlink(missing_ok=True)
    print(f"{run_path.name}: {len(combined)} rows x {len(combined.columns)} metrics, "
          f"{len(payload)} aggregate blocks{' (legacy removed)' if remove else ''}")
    return run_path / PER_EXAMPLE_CSV


def migrate_all(root: str, remove: bool = False) -> None:
    """Migrate every run under `root` that still has the legacy files.

    Args:
        root: Directory to search, e.g. the medleymd edit outputs.
        remove: Delete each run's legacy files once verified.
    """
    runs = sorted({path.parent for path in Path(root).rglob("lpaps_to_source.csv")})
    print(f"{len(runs)} run(s) to migrate under {root}")
    migrated = 0
    for run in runs:
        try:
            migrate(str(run), remove=remove)
            migrated += 1
        except (FileNotFoundError, RuntimeError, AssertionError) as exc:
            print(f"SKIP {run}: {exc}")
    print(f"\nmigrated {migrated}/{len(runs)} runs"
          + (f", {migrated * 5} inodes freed" if remove else ", legacy kept (pass --remove True)"))


if __name__ == "__main__":
    fire.Fire({"migrate": migrate, "migrate_all": migrate_all})
