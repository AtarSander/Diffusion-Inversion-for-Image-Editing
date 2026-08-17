# ABOUTME: Find edit outputs that are shorter than their track's other outputs, which is what a
# ABOUTME: job killed mid-write leaves behind, and print the files to delete before resuming.

from collections import defaultdict
from pathlib import Path

import fire
import torchaudio

from editing.AudioEditingCode.code.env import PATH_AUDIOS_MEDLEY, PATH_PROMPTS_MEDLEY
from editing.dataset_medley import prepare_dataset


def main(run_dir: str, tolerance: float = 0.98, unique_tracks: bool = False) -> None:
    """Report outputs whose duration falls short of others generated from the same source.

    Every source track appears many times in the benchmark, and each occurrence is truncated the
    same way, so all outputs from one track should have the same length. A file killed part-way
    through `torchaudio.save` is shorter, and nothing in the file itself reveals that: torchaudio
    reports the frames present, so it looks like a legitimately shorter clip.

    Args:
        run_dir: Directory of `a{idx}.wav` outputs.
        tolerance: Flag files below this fraction of their track's longest output.
        unique_tracks: Set if the run used the 35-track subset, where each track appears once and
            no within-track comparison is possible.
    """
    frame = prepare_dataset(
        Path(PATH_AUDIOS_MEDLEY), Path(PATH_PROMPTS_MEDLEY), unique_tracks=unique_tracks
    )
    by_track: dict[str, list[tuple[int, float]]] = defaultdict(list)
    missing = []
    for idx, row in frame.iterrows():
        path = Path(run_dir) / f"a{idx}.wav"
        if not path.exists():
            missing.append(idx)
            continue
        info = torchaudio.info(str(path))
        by_track[str(row["path_yt"])].append((idx, info.num_frames / info.sample_rate))

    if unique_tracks:
        print("unique_tracks: one output per track, so length outliers cannot be detected here.")

    print(f"{len(frame)} rows, {len(missing)} missing, {len(by_track)} distinct tracks\n")
    suspects = []
    for track, entries in sorted(by_track.items()):
        longest = max(duration for _, duration in entries)
        for idx, duration in entries:
            if duration < tolerance * longest:
                suspects.append((idx, duration, longest, Path(track).name))

    if not suspects:
        print("No short outputs found: every file matches its track's length.")
    else:
        print(f"{len(suspects)} suspect file(s) -- shorter than others from the same track:\n")
        for idx, duration, longest, name in sorted(suspects):
            print(f"  a{idx}.wav  {duration:7.2f}s  vs {longest:7.2f}s  ({name})")
        print("\nDelete these before resuming with --skip_existing, or they will be kept:")
        print("  rm " + " ".join(f"{run_dir}/a{idx}.wav" for idx, _, _, _ in sorted(suspects)))

    if missing:
        print(f"\n{len(missing)} row(s) have no output yet: a{missing[0]}.wav ... "
              f"a{missing[-1]}.wav")


if __name__ == "__main__":
    fire.Fire(main)
