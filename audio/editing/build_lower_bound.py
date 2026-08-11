# ABOUTME: Build the paired per-example reference ("lower bound") that eval_medley.py compares
# ABOUTME: edits against for FAD and mel PSNR/SSIM, by copying each row's MedleyDB source mix.

import shutil
from pathlib import Path

import fire
import pandas as pd

from editing.AudioEditingCode.code.env import PATH_AUDIOS_MEDLEY, PATH_PROMPTS_MEDLEY

# Splits produced by notebooks/08_create_medley_small.ipynb, stratified by `edit`.
SPLIT_CSVS = {
    "full": "captions_gpt5.csv",
    "hparam": "captions_gpt5_hparam.csv",
    "test": "captions_gpt5_test.csv",
    "loc": "captions_gpt5_loc.csv",
}


def build_split(split: str, out_root: Path, overwrite: bool = False) -> Path:
    """Copy one source mix per benchmark row, named to match the edit outputs.

    The name `a{idx}.wav` uses the row's position in the split, exactly as the edit drivers
    number their outputs. `get_filename_intersection_ratio` needs >99% filename overlap or
    `calculate_psnr_ssim` silently returns -1 instead of failing.

    Args:
        split: Key of `SPLIT_CSVS`.
        out_root: Directory that will hold `<split>/audios/a{idx}.wav`.
        overwrite: Re-copy files that already exist.

    Returns:
        The directory the wavs were written to.
    """
    csv_path = Path(PATH_PROMPTS_MEDLEY).parent / SPLIT_CSVS[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, header=0)
    target_dir = out_root / f"lower_bound_{split}" / "audios"
    target_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        filename = row["filename"]
        dirname = filename.split("_MIX")[0]
        source = (Path(PATH_AUDIOS_MEDLEY) / dirname / filename).resolve()
        if not source.exists():
            raise FileNotFoundError(f"{csv_path.name} row {idx}: missing source {source}")
        target = target_dir / f"a{idx}.wav"
        if target.exists() and not overwrite:
            continue
        shutil.copy(source, target)
        written += 1

    produced = sorted(target_dir.glob("a*.wav"))
    if len(produced) != len(df):
        raise RuntimeError(
            f"{split}: expected {len(df)} reference wavs, found {len(produced)} in {target_dir}"
        )
    print(f"{split:7s} {len(df):4d} rows -> {target_dir}  ({written} copied, "
          f"{len(df) - written} already present)")
    return target_dir


def main(
    splits: str | tuple[str, ...] = "full,hparam,test,loc",
    out_root: str | None = None,
    overwrite: bool = False,
):
    """Build reference sets for the given splits.

    Args:
        splits: Comma-separated subset of full,hparam,test,loc. Fire turns a comma-separated
            argument into a tuple, so both forms are accepted.
        out_root: Destination root; defaults to `audio/outputs/medleymd`.
        overwrite: Re-copy files that already exist.
    """
    root = (
        Path(out_root).resolve()
        if out_root
        else (Path(__file__).resolve().parents[1] / "outputs/medleymd")
    )
    names = splits.split(",") if isinstance(splits, str) else list(splits)
    for split in [str(s).strip() for s in names if str(s).strip()]:
        if split not in SPLIT_CSVS:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_CSVS)}")
        build_split(split, root, overwrite=overwrite)


if __name__ == "__main__":
    fire.Fire(main)
