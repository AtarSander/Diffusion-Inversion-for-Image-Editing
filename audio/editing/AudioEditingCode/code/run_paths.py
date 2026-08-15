# ABOUTME: Build the output directory for an edit run, refusing the empty or escaping components
# ABOUTME: that would silently scatter results outside the run's own folder.

from pathlib import Path


def resolve_run_dir(outputs_root: str | Path, *parts: str) -> Path:
    """Resolve `<outputs_root>/<parts...>/audios`, validating every component.

    `Path("a") / "" / "audios"` collapses to `a/audios` without complaint, so an empty run name
    sends every shard into the parent directory shared with other runs. An absolute or
    `..`-containing component escapes the outputs tree entirely. Both are silent, and both are
    only noticed once results are missing or mixed together, so they are rejected here.

    Args:
        outputs_root: Root for edit outputs, i.e. `PATH_EDIT_OUTPUTS`.
        *parts: Directory components, e.g. dataset, method, run name.

    Returns:
        The `audios` directory for this run, absolute.

    Raises:
        ValueError: If a component is empty, whitespace, or would escape `outputs_root`.
    """
    root = Path(outputs_root).resolve()
    for index, part in enumerate(parts):
        if part is None or not str(part).strip():
            raise ValueError(
                f"Empty path component at position {index} in {parts!r}. An empty run name "
                "would put this run's audio in the parent directory, mixed with other runs."
            )
        if Path(str(part)).is_absolute() or ".." in Path(str(part)).parts:
            raise ValueError(f"Path component {part!r} would escape {root}")

    run_dir = root.joinpath(*(str(p) for p in parts), "audios").resolve()
    if root not in run_dir.parents:
        raise ValueError(f"{run_dir} is not inside {root}")
    return run_dir
