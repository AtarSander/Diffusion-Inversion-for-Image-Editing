import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from editing.AudioEditingCode.evals.lpaps import LPAPS
from editing.AudioEditingCode.evals.meta_clap_consistency import CLAPTextConsistencyMetric
from editing.AudioEditingCode.evals.utils import calc_clap_win, calc_lpaps_win
from src.preprocess.utils import extract_prompts_csv

# download music_audioset_epoch_15_esc_90.14.pt to res/clap/pretrained dir from https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt


def load_audios_from_path(path: Path | str) -> torch.Tensor:
    """Loads audios from a numpy file and saves them as wav files."""

    if isinstance(path, str):
        path = Path(path)
    assert path.suffix == ".npy"

    if not path.exists():
        # find all .npy files that are named xxx_{int}.npy
        npy_files = list(path.parent.glob(path.stem + "_*.npy"))
        npy_files = sorted(npy_files, key=lambda x: int(x.stem.split("_")[-1]))
    else:
        npy_files = [path]

    all_audios = []
    for npy_path in npy_files:
        audios = np.load(npy_path)
        audios_tensor = torch.from_numpy(audios)
        if len(audios_tensor.shape) == 2:
            audios_tensor = audios_tensor.unsqueeze(1)
        all_audios.append(audios_tensor)
    all_audios = torch.cat(all_audios, dim=0)
    return all_audios


def main(
    feature: str,
    run_patch: str,
    prompts_path: str,
    n_latents_pp: int,
    sampling_rate: int,
    output_dir: str | None = None,
):
    if output_dir is None:
        dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"editing/outputs/ace/{dt_str}_{feature}_{run_patch}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    paths_audios = {
        "target_prompt_audio": f"outputs/ace/patching/{feature}/none/audios/clean.npy",
        "source_prompt_audio": f"outputs/ace/patching/{feature}/none/audios/patched.npy",
        "ours_patching_audio": f"outputs/ace/patching/{feature}/{run_patch}/audios/patched.npy",
    }
    target_prompt_audio = load_audios_from_path(paths_audios["target_prompt_audio"])
    source_prompt_audio = load_audios_from_path(paths_audios["source_prompt_audio"])
    ours_patching_audio = load_audios_from_path(paths_audios["ours_patching_audio"])
    target_prompts, source_prompts = extract_prompts_csv(prompts_path, feature=feature)
    target_prompts = [p for p in target_prompts for _ in range(n_latents_pp)]
    source_prompts = [p for p in source_prompts for _ in range(n_latents_pp)]

    assert (
        target_prompt_audio.shape[0]
        == source_prompt_audio.shape[0]
        == ours_patching_audio.shape[0]
        == len(target_prompts)
        == len(source_prompts)
    ), f"shapes are incorrect; target_prompt_audio.shape[0]: {target_prompt_audio.shape[0]}, source_prompt_audio.shape[0]: {source_prompt_audio.shape[0]}, ours_patching_audio.shape[0]: {ours_patching_audio.shape[0]}, len(target_prompts): {len(target_prompts)}, len(source_prompts): {len(source_prompts)}"

    # calculate metrics - initialize models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # LPAPS
    lpaps_model = LPAPS(
        net="clap",
        device=device,
        net_kwargs={
            "model_arch": "HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
            "chkpt": "music_audioset_epoch_15_esc_90.14.pt",
            "enable_fusion": "fusion" in "music_audioset_epoch_15_esc_90.14.pt",
        },
        checkpoint_path="res/clap/pretrained",
    )
    lpaps_source_source = {}
    lpaps_source_target = {}
    lpaps_source_ours = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(source_prompt_audio)), desc="Calculating LPAPS"):
            lpaps_source_source[audio_idx] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_prompt_audio[audio_idx],
                aud2=source_prompt_audio[audio_idx],
                sr1=sampling_rate,
                sr2=sampling_rate,
                win_length=(10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None),
                overlap=0.1,
                method="mean",
                device=device,
            )
            lpaps_source_target[audio_idx] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_prompt_audio[audio_idx],
                aud2=target_prompt_audio[audio_idx],
                sr1=sampling_rate,
                sr2=sampling_rate,
                win_length=(10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None),
                overlap=0.1,
                method="mean",
                device=device,
            )
            lpaps_source_ours[audio_idx] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_prompt_audio[audio_idx],
                aud2=ours_patching_audio[audio_idx],
                sr1=sampling_rate,
                sr2=sampling_rate,
                win_length=(10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None),
                overlap=0.1,
                method="mean",
                device=device,
            )
    lpaps_source_source_df = pd.DataFrame(list(lpaps_source_source.items()), columns=["audio_idx", "lpaps"])
    lpaps_source_target_df = pd.DataFrame(list(lpaps_source_target.items()), columns=["audio_idx", "lpaps"])
    lpaps_source_ours_df = pd.DataFrame(list(lpaps_source_ours.items()), columns=["audio_idx", "lpaps"])
    lpaps_source_source_df.to_csv(os.path.join(output_dir, "lpaps_source_source.csv"))
    lpaps_source_target_df.to_csv(os.path.join(output_dir, "lpaps_source_target.csv"))
    lpaps_source_ours_df.to_csv(os.path.join(output_dir, "lpaps_source_ours.csv"))
    print(f"LPAPS saved to {output_dir}")

    # CLAP
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)
    clap_model.eval()

    clap_source_sourcep = {}
    clap_source_targetp = {}
    clap_target_sourcep = {}
    clap_target_targetp = {}
    clap_ours_sourcep = {}
    clap_ours_targetp = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(source_prompt_audio)), desc="Calculating CLAP"):
            sp = source_prompts[audio_idx]
            tp = target_prompts[audio_idx]

            clap_source_sourcep[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=source_prompt_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=sp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": sp,
            }
            clap_source_targetp[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=source_prompt_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=tp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": tp,
            }
            clap_target_sourcep[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=target_prompt_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=sp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": sp,
            }
            clap_target_targetp[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=target_prompt_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=tp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": tp,
            }
            clap_ours_sourcep[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=ours_patching_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=sp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": sp,
            }
            clap_ours_targetp[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=ours_patching_audio[audio_idx],
                    sr=sampling_rate,
                    target_prompt=tp,
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": tp,
            }

    # save CLAP
    clap_source_sourcep_df = pd.DataFrame.from_dict(clap_source_sourcep, orient="index")
    clap_source_targetp_df = pd.DataFrame.from_dict(clap_source_targetp, orient="index")
    clap_target_sourcep_df = pd.DataFrame.from_dict(clap_target_sourcep, orient="index")
    clap_target_targetp_df = pd.DataFrame.from_dict(clap_target_targetp, orient="index")
    clap_ours_sourcep_df = pd.DataFrame.from_dict(clap_ours_sourcep, orient="index")
    clap_ours_targetp_df = pd.DataFrame.from_dict(clap_ours_targetp, orient="index")
    clap_source_sourcep_df.to_csv(os.path.join(output_dir, "clap_source_sourcep.csv"))
    clap_source_targetp_df.to_csv(os.path.join(output_dir, "clap_source_targetp.csv"))
    clap_target_sourcep_df.to_csv(os.path.join(output_dir, "clap_target_sourcep.csv"))
    clap_target_targetp_df.to_csv(os.path.join(output_dir, "clap_target_targetp.csv"))
    clap_ours_sourcep_df.to_csv(os.path.join(output_dir, "clap_ours_sourcep.csv"))
    clap_ours_targetp_df.to_csv(os.path.join(output_dir, "clap_ours_targetp.csv"))
    print(f"CLAP saved to {output_dir}")

    # final results
    final_results = {
        "lpaps_source_source": {
            "mean": lpaps_source_source_df["lpaps"].mean(),
            "std": lpaps_source_source_df["lpaps"].std(),
        },
        "lpaps_source_target": {
            "mean": lpaps_source_target_df["lpaps"].mean(),
            "std": lpaps_source_target_df["lpaps"].std(),
        },
        "lpaps_source_ours": {
            "mean": lpaps_source_ours_df["lpaps"].mean(),
            "std": lpaps_source_ours_df["lpaps"].std(),
        },
        "clap_source_sourcep": {
            "mean": clap_source_sourcep_df["clap"].mean(),
            "std": clap_source_sourcep_df["clap"].std(),
        },
        "clap_source_targetp": {
            "mean": clap_source_targetp_df["clap"].mean(),
            "std": clap_source_targetp_df["clap"].std(),
        },
        "clap_target_sourcep": {
            "mean": clap_target_sourcep_df["clap"].mean(),
            "std": clap_target_sourcep_df["clap"].std(),
        },
        "clap_target_targetp": {
            "mean": clap_target_targetp_df["clap"].mean(),
            "std": clap_target_targetp_df["clap"].std(),
        },
        "clap_ours_sourcep": {
            "mean": clap_ours_sourcep_df["clap"].mean(),
            "std": clap_ours_sourcep_df["clap"].std(),
        },
        "clap_ours_targetp": {
            "mean": clap_ours_targetp_df["clap"].mean(),
            "std": clap_ours_targetp_df["clap"].std(),
        },
    }
    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv(os.path.join(output_dir, "final_results.csv"))
    print("Final results:", df_final_results)
    print(f"Final results saved to {output_dir}/final_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--run_patch", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--n_latents_pp", type=int, required=True)
    parser.add_argument("--sampling_rate", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default=None)
    args = parser.parse_args()

    main(
        feature=args.feature,
        run_patch=args.run_patch,
        prompts_path=args.prompts_path,
        n_latents_pp=args.n_latents_pp,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
    )

# prompts_path = "data/drums/corrupted_prompts_small.csv"
# n_latents_pp = 8
