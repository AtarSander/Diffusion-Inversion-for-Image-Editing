# ABOUTME: Shared MedleyMD benchmark dataframe builder, kept free of model imports so the eval
# ABOUTME: scripts do not have to load ACE-Step just to read the prompt CSV.

from pathlib import Path

import pandas as pd


def prepare_dataset(
    path_audios: Path, path_prompts: Path, unique_tracks: bool = False
) -> pd.DataFrame:
    """Join the MedleyMD prompt CSV with the MedleyDB mixes it refers to.

    Row order defines the `a{idx}.wav` numbering used by every edit driver and by the paired
    reference set, so this must stay a plain positional read of the CSV.

    Args:
        path_audios: Root holding `<Track>/<Track>_MIX.wav`.
        path_prompts: Prompt CSV with filename, source_captions, target_captions, edit.
        unique_tracks: Keep only the first row per source file, giving one entry per distinct
            MedleyDB track. Row order and the `a{idx}` numbering are preserved, so the subset
            stays addressable by the same indices as the full set.

    Returns:
        One row per edit, with resolved audio path, source/target prompt and edit class.
    """
    df_prompts = pd.read_csv(path_prompts, index_col=0, header=0)
    if unique_tracks:
        df_prompts = df_prompts.drop_duplicates(subset="filename", keep="first")
    rows = []
    for _, row in df_prompts.iterrows():
        filename = row["filename"]
        dirname = filename.split("_MIX")[0]
        rows.append(
            {
                "path_yt": (Path(path_audios) / dirname / filename).resolve(),
                "original_prompt": row["source_captions"],
                "editing_prompt": row["target_captions"],
                "edit_class": row["edit"],
            }
        )
    return pd.DataFrame(rows, index=df_prompts.index)
