import json
import os
import shutil
from pathlib import Path

import fire
import pandas as pd
import torch
import torchaudio
from muq import MuQMuLan
from torchaudio.transforms import Resample
from tqdm import tqdm

from editing.AudioEditingCode.evals.lpaps import LPAPS
from editing.AudioEditingCode.evals.meta_clap_consistency import CLAPTextConsistencyMetric, convert_audio
from editing.AudioEditingCode.evals.utils import calc_clap_win, calc_lpaps_win
from editing.dataset_medley import prepare_dataset
from editing.AudioEditingCode.code.env import PATH_AUDIOS_MEDLEY, medley_split_paths
from src.metrics.alignment import MusicAlignmentEval
DISABLE_TQDM = False



def prepare_data(
    path_edited_audio: str,
    limit: int | None = None,
    unique_tracks: bool = False,
    split: str = "full",
):
    df_musiccaps = prepare_dataset(
        Path(PATH_AUDIOS_MEDLEY),
        Path(medley_split_paths(split)[0]),
        unique_tracks=unique_tracks,
    )
    if limit is not None:
        df_musiccaps = df_musiccaps.head(limit)
    target_prompts = []
    source_prompts = []
    source_audios = []
    edits = []
    srs_src = []
    srs_edit = []
    classification_task = []
    for idx, row in df_musiccaps.iterrows():
        classification_task.append(row["edit_class"])
        target_prompts.append(row["editing_prompt"])
        source_prompts.append(row["original_prompt"])
        source_audio, source_audio_sr = torchaudio.load(row["path_yt"])
        edit_audio, edit_audio_sr = torchaudio.load(path_edited_audio + f"/a{idx}.wav")
        # assert (
        #     source_audio_sr == edit_audio_sr
        # ), f"Track {row['path_yt']} (sr={source_audio_sr}) has different sample rate than edited audio a{idx}.wav (sr={edit_audio_sr})"
        # if edit_audio.shape[1] < source_audio.shape[1]:
        #     source_audio = source_audio[:, : edit_audio.shape[1]]
        source_audios.append(source_audio)
        edits.append(edit_audio)
        srs_src.append(source_audio_sr)
        srs_edit.append(edit_audio_sr)
    return target_prompts, source_prompts, source_audios, edits, srs_src, srs_edit, classification_task


def load_models(device):
    # calculate metrics - initialize models
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
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)

    clap_model = clap_model.eval()

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    return lpaps_model, clap_model, mulan


def get_lpaps(source_audios, edits, srs_src, srs_edit, device):
    # load lpaps
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

    # process
    lpaps_source_target = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(source_audios)), desc="Calculating LPAPS", disable=DISABLE_TQDM):
            lpaps_source_target[audio_idx] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_audios[audio_idx],
                aud2=edits[audio_idx],
                sr1=srs_src[audio_idx],
                sr2=srs_edit[audio_idx],
                win_length=(10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None),
                overlap=0.1,
                method="mean",
                device=device,
            )
    lpaps_source_target_df = pd.DataFrame(list(lpaps_source_target.items()), columns=["audio_idx", "lpaps"])
    return lpaps_source_target_df


def get_clap(target_prompts, edits, srs_edit, device):
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)
    clap_model = clap_model.eval()

    clap_target_targetp = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(edits)), desc="Calculating CLAP", disable=DISABLE_TQDM):
            clap_target_targetp[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=edits[audio_idx],
                    sr=srs_edit[audio_idx],
                    target_prompt=target_prompts[audio_idx],
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": target_prompts[audio_idx],
            }
    clap_target_targetp_df = pd.DataFrame.from_dict(clap_target_targetp, orient="index")
    return clap_target_targetp_df


def get_mulan(target_prompts, edits, srs_edit, device):
    # init mulan

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    mulan_target_targetp = {}
    with torch.no_grad():
        all_similarities = []
        for audio_idx in tqdm(range(len(edits)), desc="Calculating MUQT", disable=DISABLE_TQDM):
            all_texts = [target_prompts[audio_idx]]
            text_embeds = mulan(texts=all_texts)
            batch_audio = edits[audio_idx]
            batch_audio = Resample(srs_edit[audio_idx], 24000)(batch_audio)
            # MuQ-MuLan iterates dim 0 as the batch, so a stereo edit yields two latents and
            # shifts every later row of `similarities` against its prompt. Downmix to mono, as
            # the CLAP path already does via convert_audio(..., to_channels=1).
            batch_wavs = batch_audio.mean(dim=0, keepdim=True).to(mulan.device)
            batch_embed = mulan(wavs=batch_wavs)
            batch_similarities = mulan.calc_similarity(batch_embed, text_embeds)
            all_similarities.append(batch_similarities.cpu())
        similarities = torch.cat(all_similarities, dim=0)
        assert similarities.shape == (len(edits), 1), (
            f"expected one similarity per edit, got {tuple(similarities.shape)} for "
            f"{len(edits)} edits: the audio batch dimension is not 1"
        )

        per_prompt_sims = {}
        for audio_idx in range(len(edits)):
            per_prompt_sims[audio_idx] = {}
            p_idx = 0
            per_prompt_sims[audio_idx][f"muqt_sim_p{p_idx}"] = similarities[audio_idx, p_idx].item()
            per_prompt_sims[audio_idx][f"p{p_idx}"] = target_prompts[audio_idx]

    mulan_target_targetp_df = pd.DataFrame.from_dict(per_prompt_sims, orient="index")
    return mulan_target_targetp_df

def directional_similarity(
    src_audio_emb: torch.Tensor,
    edit_audio_emb: torch.Tensor,
    src_text_emb: torch.Tensor,
    tgt_text_emb: torch.Tensor,
    w: float = 1.0,
) -> float:
    """Cosine between the audio edit direction and the caption edit direction.

    The audio-domain analogue of directional CLIP (StyleGAN-NADA): it asks whether the change
    the edit made to the audio points the same way as the change from source to target caption,
    which a copy of the input scores 0 on however well it matches the target prompt on its own.

    Args:
        src_audio_emb: Embedding of the source audio, `[D]`.
        edit_audio_emb: Embedding of the edited audio, `[D]`.
        src_text_emb: Embedding of the source caption, `[D]`.
        tgt_text_emb: Embedding of the target caption, `[D]`.
        w: Scaling factor. 1.0 keeps the value a plain cosine in [-1, 1].

    Returns:
        Directional similarity, higher is better.
    """
    for name, emb in (
        ("src_audio_emb", src_audio_emb),
        ("edit_audio_emb", edit_audio_emb),
        ("src_text_emb", src_text_emb),
        ("tgt_text_emb", tgt_text_emb),
    ):
        assert emb.ndim == 1, f"{name} must be [D], got {tuple(emb.shape)}"
    assert src_audio_emb.shape == edit_audio_emb.shape, "audio embeddings differ in width"
    assert src_text_emb.shape == tgt_text_emb.shape, "text embeddings differ in width"

    delta_audio = torch.nn.functional.normalize(edit_audio_emb - src_audio_emb, dim=-1, eps=1e-6)
    delta_text = torch.nn.functional.normalize(tgt_text_emb - src_text_emb, dim=-1, eps=1e-6)
    return w * torch.dot(delta_audio.float(), delta_text.float()).item()


def clap_audio_embedding(clap_model, aud, sr, device, win_length: int = 10, overlap: float = 0.1):
    """Clip-level CLAP audio embedding: mean of L2-normalised window embeddings.

    The non-fusion checkpoint consumes 10 s at a time, which is why `calc_clap_win` scores in
    windows; this mirrors that segmentation so the directional score sees the same audio the
    CLAP score does.

    Args:
        clap_model: A `CLAPTextConsistencyMetric` wrapping the LAION CLAP module.
        aud: Waveform `[C, T]`.
        sr: Sample rate of `aud`.
        device: Torch device.
        win_length: Window length in seconds.
        overlap: Fractional overlap between consecutive windows.

    Returns:
        Embedding `[D]`.
    """
    window_samples = int(sr * win_length)
    hop = int(window_samples * (1 - overlap))
    embeddings = []
    for start in range(0, aud.shape[-1], hop):
        window = aud[:, start : start + window_samples]
        if window.shape[-1] < sr:  # a sub-second tail carries no usable content
            continue
        wav = convert_audio(
            window.unsqueeze(0).to(device), from_rate=sr, to_rate=48_000, to_channels=1
        ).mean(dim=1)
        embedding = clap_model.model.get_audio_embedding_from_data(wav, use_tensor=True)
        embeddings.append(torch.nn.functional.normalize(embedding, dim=-1))
    assert embeddings, f"no window of at least 1 s in a {aud.shape[-1] / sr:.2f} s clip"
    return torch.cat(embeddings, dim=0).mean(dim=0)


def get_directional(
    source_prompts, target_prompts, source_audios, edits, srs_src, srs_edit, device, w: float = 1.0
):
    """Directional CLAP and MuLan scores for every edit.

    The source is truncated to the edit's duration first: the edit drivers truncate their input
    (60 s for AudioLDM2, the transformer's sample_size for Stable Audio), so embedding the whole
    multi-minute mix would compare different stretches of music.

    Args:
        source_prompts: Caption describing each source clip.
        target_prompts: Caption describing each desired edit.
        source_audios: Source waveforms `[C, T]`.
        edits: Edited waveforms `[C, T]`.
        srs_src: Sample rate per source waveform.
        srs_edit: Sample rate per edited waveform.
        device: Torch device.
        w: Scaling factor passed to `directional_similarity`.

    Returns:
        DataFrame indexed by audio index with `clap_dir` and `mulan_dir` columns.
    """
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)
    clap_model = clap_model.eval()

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    rows = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(edits)), desc="Calculating directional", disable=DISABLE_TQDM):
            edit = edits[audio_idx]
            sr_edit, sr_src = srs_edit[audio_idx], srs_src[audio_idx]
            edit_seconds = edit.shape[-1] / sr_edit
            source = source_audios[audio_idx][:, : int(edit_seconds * sr_src)]

            clap_text = clap_model.model.get_text_embedding(
                [source_prompts[audio_idx], target_prompts[audio_idx]],
                tokenizer=clap_model._tokenizer,
                use_tensor=True,
            )
            clap_dir = directional_similarity(
                src_audio_emb=clap_audio_embedding(clap_model, source, sr_src, device),
                edit_audio_emb=clap_audio_embedding(clap_model, edit, sr_edit, device),
                src_text_emb=clap_text[0],
                tgt_text_emb=clap_text[1],
                w=w,
            )

            mulan_text = mulan(texts=[source_prompts[audio_idx], target_prompts[audio_idx]])
            mulan_source = Resample(sr_src, 24000)(source).mean(dim=0, keepdim=True).to(mulan.device)
            mulan_edit = Resample(sr_edit, 24000)(edit).mean(dim=0, keepdim=True).to(mulan.device)
            mulan_dir = directional_similarity(
                src_audio_emb=mulan(wavs=mulan_source)[0],
                edit_audio_emb=mulan(wavs=mulan_edit)[0],
                src_text_emb=mulan_text[0],
                tgt_text_emb=mulan_text[1],
                w=w,
            )

            rows[audio_idx] = {
                "clap_dir": clap_dir,
                "mulan_dir": mulan_dir,
                "source_prompt": source_prompts[audio_idx],
                "target_prompt": target_prompts[audio_idx],
            }
            if audio_idx == 0:
                print(f"first directional scores: clap_dir={clap_dir:.4f} mulan_dir={mulan_dir:.4f}")

    return pd.DataFrame.from_dict(rows, orient="index")


def resample_audios(path_audio_orignal: Path, path_audio_resampled: Path, target_sr: int):
    for file in tqdm(path_audio_orignal.glob("*.wav"), desc=f"Resampling audios in {path_audio_orignal}"):
        audio, sr = torchaudio.load(file)
        audio = Resample(sr, target_sr)(audio)
        torchaudio.save(path_audio_resampled / file.name, audio, target_sr)


def ensure_resampled(source_dir: Path, target_sr: int) -> Path:
    """Return a `<name>_32k` sibling of `source_dir`, building it once and atomically.

    Array tasks share the reference directory, so a plain `if not exists: resample` lets one
    task create the directory while the others observe it half filled and score against a
    partial set. Building into a private temporary directory and renaming means the cache only
    ever becomes visible complete.

    Args:
        source_dir: Directory of wavs to resample.
        target_sr: Sample rate the metrics run at.

    Returns:
        Directory holding the resampled copies.
    """
    expected = len(list(source_dir.glob("*.wav")))
    if expected == 0:
        raise FileNotFoundError(f"No wavs to resample in {source_dir}")

    resampled_dir = source_dir.parent / f"{source_dir.name}_{target_sr // 1000}k"
    if resampled_dir.exists():
        found = len(list(resampled_dir.glob("*.wav")))
        if found == expected:
            return resampled_dir
        # A leftover partial cache, e.g. from an interrupted run. Drop it rather than scoring
        # against it or demanding manual cleanup.
        print(f"Rebuilding incomplete cache {resampled_dir} ({found}/{expected} wavs)")
        shutil.rmtree(resampled_dir, ignore_errors=True)

    staging = source_dir.parent / f".{source_dir.name}_{target_sr // 1000}k.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    resample_audios(source_dir, staging, target_sr)

    try:
        staging.rename(resampled_dir)
    except OSError:
        # Another task finished first; its directory is complete, so use it and drop ours.
        shutil.rmtree(staging, ignore_errors=True)

    found = len(list(resampled_dir.glob("*.wav")))
    if found != expected:
        raise RuntimeError(
            f"Resampled cache {resampled_dir} has {found} wavs but {source_dir} has {expected}. "
            "Delete it and re-run."
        )
    return resampled_dir

def calculate_source_distance_metrics(device: torch.device, path_edited_audio: str, path_lower_bound: str):
    """Score the edits against the per-example source reference in the mel domain.

    Args:
        device: Torch device.
        path_edited_audio: Directory of `a{idx}.wav` edits.
        path_lower_bound: Directory of the paired source references.

    Returns:
        The aggregate metrics (with `psnr_sem`/`ssim_sem` added) and the per-file psnr/ssim frame.
    """
    path_edited_audio_resampled = ensure_resampled(Path(path_edited_audio).resolve(), 32000)
    path_lower_bound_resampled = ensure_resampled(Path(path_lower_bound).resolve(), 32000)

    # get_filename_intersection_ratio() needs >99% overlap; below that it sets same_name=False
    # and calculate_psnr_ssim() returns -1 instead of raising. Catch the mismatch here, where
    # the cause (wrong lower-bound split, or an incomplete edit run) is still obvious.
    edited_names = {p.name for p in path_edited_audio_resampled.glob("*.wav")}
    reference_names = {p.name for p in path_lower_bound_resampled.glob("*.wav")}
    if edited_names != reference_names:
        raise ValueError(
            f"Edit/reference filenames differ: {len(edited_names)} edits vs "
            f"{len(reference_names)} references, {len(edited_names & reference_names)} shared. "
            f"Only in edits: {sorted(edited_names - reference_names)[:3]}; only in reference: "
            f"{sorted(reference_names - edited_names)[:3]}. PATH_LOWER_BOUND_MEDLEY must match "
            "the split used for PATH_PROMPTS_MEDLEY."
        )

    evaluator = MusicAlignmentEval(sampling_rate=32000, device=device)
    metrics = evaluator.main(
        generate_files_path=str(path_edited_audio_resampled),
        groundtruth_path=str(path_lower_bound_resampled),
        limit_num=None,
    )
    for key in ("psnr", "ssim"):
        if str(metrics.get(key)).startswith("-1"):
            raise ValueError(
                f"{key}={metrics[key]} is the sentinel for a paired-metric failure, not a score."
            )

    per_file = pd.DataFrame.from_dict(metrics.pop("psnr_ssim_per_file"), orient="index")
    per_file.index.name = "filename"
    assert len(per_file) == len(edited_names), (
        f"{len(per_file)} per-file psnr/ssim rows for {len(edited_names)} edits: "
        "some pairs were dropped, so the mean and its error cover different sets"
    )
    metrics["psnr_sem"] = f"{per_file['psnr'].sem():.3f}"
    metrics["ssim_sem"] = f"{per_file['ssim'].sem():.3f}"
    return metrics, per_file


def main(
    path_audio: str,
    limit: int | None = None,
    unique_tracks: bool = False,
    split: str = "full",
):
    """Score one edit run against its source audio and its target prompts.

    Args:
        path_audio: Directory of `a{idx}.wav` edits to score.
        limit: Score only the first N examples and skip the directory-level PSNR/SSIM pass, to
            check the wiring in seconds. The written files are then partial, not a result.
        unique_tracks: Score the 35-track subset instead of all 696 rows. The paired reference
            must have been built with the same flag.
        split: Benchmark split the run was produced with, i.e. which prompt CSV names its rows.
            Must match the driver's --split, or the prompts are paired with the wrong audio.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (
        target_prompts,
        source_prompts,
        source_audios,
        edits,
        srs_src,
        srs_edit,
        classification_tasks,
    ) = prepare_data(path_audio, limit=limit, unique_tracks=unique_tracks, split=split)
    if limit is not None:
        print(f"*** SMOKE RUN: first {len(edits)} examples only, PSNR/SSIM skipped ***")
    path_save_metrics = Path(path_audio).parent
    lpaps_source_target_df = get_lpaps(source_audios, edits, srs_src, srs_edit, device)
    lpaps_source_target_df["classification_task"] = classification_tasks
    lpaps_source_target_df.to_csv((path_save_metrics / "lpaps_to_source.csv"))
    clap_target_targetp_df = get_clap(target_prompts, edits, srs_edit, device)
    clap_target_targetp_df["classification_task"] = classification_tasks
    clap_target_targetp_df.to_csv((path_save_metrics / "clap_to_target_prompt.csv"))
    mulan_target_targetp_df = get_mulan(target_prompts, edits, srs_edit, device)
    mulan_target_targetp_df["classification_task"] = classification_tasks
    mulan_target_targetp_df.to_csv((path_save_metrics / "mulan_to_target_prompt.csv"))
    directional_df = get_directional(
        source_prompts, target_prompts, source_audios, edits, srs_src, srs_edit, device
    )
    directional_df["classification_task"] = classification_tasks
    directional_df.to_csv((path_save_metrics / "directional_to_prompts.csv"))

    final_results = {
        "LPAPS": {
            "mean": lpaps_source_target_df["lpaps"].mean(),
            "std": lpaps_source_target_df["lpaps"].std(),
        },
        "CLAP": {
            "mean": clap_target_targetp_df["clap"].mean(),
            "std": clap_target_targetp_df["clap"].std(),
        },
        "MUQT": {
            "mean": mulan_target_targetp_df["muqt_sim_p0"].mean(),
            "std": mulan_target_targetp_df["muqt_sim_p0"].std(),
        },
        "CLAP_DIR": {
            "mean": directional_df["clap_dir"].mean(),
            "std": directional_df["clap_dir"].std(),
        },
        "MUQT_DIR": {
            "mean": directional_df["mulan_dir"].mean(),
            "std": directional_df["mulan_dir"].std(),
        },
    }
    print(final_results)
    with open((path_save_metrics / "final_results.json"), "w") as f:
        json.dump(final_results, f)

    per_task_result = {
        task: {
            "LPAPS": {
                "mean": lpaps_source_target_df[lpaps_source_target_df["classification_task"] == task]["lpaps"].mean(),
                "std": lpaps_source_target_df[lpaps_source_target_df["classification_task"] == task]["lpaps"].std(),
            },
            "CLAP": {
                "mean": clap_target_targetp_df[clap_target_targetp_df["classification_task"] == task]["clap"].mean(),
                "std": clap_target_targetp_df[clap_target_targetp_df["classification_task"] == task]["clap"].std(),
            },
            "MUQT": {
                "mean": mulan_target_targetp_df[mulan_target_targetp_df["classification_task"] == task][
                    "muqt_sim_p0"
                ].mean(),
                "std": mulan_target_targetp_df[mulan_target_targetp_df["classification_task"] == task][
                    "muqt_sim_p0"
                ].std(),
            },
            "CLAP_DIR": {
                "mean": directional_df[directional_df["classification_task"] == task]["clap_dir"].mean(),
                "std": directional_df[directional_df["classification_task"] == task]["clap_dir"].std(),
            },
            "MUQT_DIR": {
                "mean": directional_df[directional_df["classification_task"] == task]["mulan_dir"].mean(),
                "std": directional_df[directional_df["classification_task"] == task]["mulan_dir"].std(),
            },
        }
        for task in list(set(classification_tasks))
    }
    with open((path_save_metrics / "per_task_results.json"), "w") as f:
        json.dump(per_task_result, f)

    if limit is not None:
        print("smoke run: PSNR/SSIM need the whole directory, skipping. Re-run without --limit.")
        return

    # Every subset has its own reference; the full one shares only part of its filenames with a
    # subset's edits, and the paired metrics refuse to score that.
    lower_bound = medley_split_paths("tracks" if unique_tracks else split)[1]
    print(f"paired reference: {lower_bound}")
    source_distance_metrics, psnr_ssim_per_file = calculate_source_distance_metrics(device=device, path_edited_audio=path_audio, path_lower_bound=lower_bound)
    psnr_ssim_per_file.to_csv((path_save_metrics / "psnr_ssim_per_file.csv"))

    with open((path_save_metrics / "source_distance_metrics.json"), "w") as f:
        json.dump(source_distance_metrics, f)


if __name__ == "__main__":
    fire.Fire(main)
