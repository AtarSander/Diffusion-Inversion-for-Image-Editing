# ABOUTME: Download the MusicCaps audio for the first N caption rows (the inversion-LoRA training
# ABOUTME: slice), reusing the vendored downloader; resumes by skipping files already on disk.

import argparse
import multiprocessing as mp
import os

import pandas as pd
from utils import download_ps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True, type=str)
    # 1650 = the 1500 training rows plus margin for removed videos.
    parser.add_argument("--num", default=1650, type=int)
    parser.add_argument("--processes", default=8, type=int)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    meta = pd.read_csv("metadata/musiccaps-public.csv").head(args.num)
    missing = meta[
        ~meta.apply(
            lambda r: os.path.exists(
                f"{args.save_path}/[{r.ytid}]-[{int(r.start_s)}-{int(r.end_s)}].wav"
            ),
            axis=1,
        )
    ]
    print(f"{len(meta)} rows in slice, {len(missing)} to download", flush=True)
    if len(missing):
        download_ps(
            missing["ytid"],
            (missing.start_s * 1000).astype(int),
            (missing.end_s * 1000).astype(int),
            args.save_path,
            "audio",
            num_processes=args.processes,
        )
    have = len([f for f in os.listdir(args.save_path) if f.endswith(".wav")])
    print(f"done: {have} wavs in {args.save_path}", flush=True)
