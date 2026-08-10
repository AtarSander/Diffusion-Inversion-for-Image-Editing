from datetime import datetime
from pathlib import Path

import fire
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from datasets import load_dataset
from torchaudio.transforms import Resample
from tqdm import tqdm

from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from editing.AudioEditingCode.code.env import PATH_MUSICCAPS, PATH_EDIT_OUTPUTS



EDITING_TYPE_ID_DEFAULT = "0"  # '5'
PATH_DIR_OUTPUT = Path(PATH_EDIT_OUTPUTS + "/zome").resolve()

dt_str = datetime.now().strftime("%Y%m%d%H%M%S")


def prepare_dataset(editing_type_id: str, path_musiccaps: Path):
    if isinstance(editing_type_id, int):
        editing_type_id = str(editing_type_id)
    ds = load_dataset("liuhuadai/ZoME-Bench")
    df_train = ds["train"].to_pandas()
    df_instruments = df_train[df_train["editing_type_id"] == editing_type_id]
    path_musiccaps = Path(path_musiccaps)
    filenames_paths = [p for p in path_musiccaps.glob("*.wav")]
    shorted_filenames = [p.name.split("]")[0][1:] for p in filenames_paths]
    name_to_path = {sfn: fnp for sfn, fnp in zip(shorted_filenames, filenames_paths)}
    df_instruments = df_instruments[df_instruments["ytid"].isin(shorted_filenames)]
    df_instruments["path_yt"] = df_instruments.apply(lambda x: name_to_path.get(x["ytid"], None), axis=1)
    return df_instruments


def init_ace(device):
    ace_step_pipeline = SimpleACEStepPipeline(device=device, dtype="bfloat16")
    ace_step_pipeline.load()
    return ace_step_pipeline


def edit_single_audio(
    path_to_audio: Path,
    source_prompt: str,
    target_prompt: str,
    ace_step_pipeline: SimpleACEStepPipeline,
    hyperparams: dict,
):
    original_audio, original_sr = torchaudio.load(path_to_audio)
    original_audio_len = original_audio.shape[1]
    edited_audio = ace_step_pipeline.edit_audio(
        src_audio_path=path_to_audio,
        source_prompt=source_prompt,
        source_lyrics=hyperparams["lyrics"],
        target_prompt=target_prompt,
        target_lyrics=hyperparams["lyrics"],
        edit_n_min=hyperparams["edit_n_min"],
        edit_n_max=hyperparams["edit_n_max"],
        edit_n_avg=hyperparams["edit_n_avg"],
        guidance_scale=hyperparams["guidance_scale"],
        use_erg_tag=hyperparams["use_erg_tag"],
        infer_step=hyperparams["infer_step"],
        oss_steps=[],
        manual_seeds=hyperparams["seed"],
    )
    resampler = Resample(hyperparams["model_sr"], original_sr)
    edited_audio_resampled = resampler(edited_audio)
    edited_audio_resampled = edited_audio_resampled[..., :original_audio_len]
    edited_audio_resampled = edited_audio_resampled.mean(dim=1)
    return edited_audio_resampled, original_sr


def edit_single_audio_hooked(
    path_to_audio: Path,
    source_prompt: str,
    target_prompt: str,
    ace_step_pipeline: SimpleACEStepPipeline,
    hyperparams: dict,
    layers_to_hook: list[str] | None = None,
):
    original_audio, original_sr = torchaudio.load(path_to_audio)
    original_audio_len = original_audio.shape[1]
    edited_audio = ace_step_pipeline.edit_audio_hooked(
        src_audio_path=path_to_audio,
        source_prompt=source_prompt,
        source_lyrics=hyperparams["lyrics"],
        target_prompt=target_prompt,
        target_lyrics=hyperparams["lyrics"],
        edit_n_min=hyperparams["edit_n_min"],
        edit_n_max=hyperparams["edit_n_max"],
        edit_n_avg=hyperparams["edit_n_avg"],
        guidance_scale=hyperparams["guidance_scale"],
        use_erg_tag=hyperparams["use_erg_tag"],
        infer_step=hyperparams["infer_step"],
        oss_steps=[],
        manual_seeds=hyperparams["seed"],
        layers_to_hook=layers_to_hook,
    )
    resampler = Resample(hyperparams["model_sr"], original_sr)
    edited_audio_resampled = resampler(edited_audio)
    edited_audio_resampled = edited_audio_resampled[..., :original_audio_len]
    edited_audio_resampled = edited_audio_resampled.mean(dim=1)
    return edited_audio_resampled, original_sr


def main(
    editing_type_id: str = EDITING_TYPE_ID_DEFAULT,
    dataset_name: str = "zome",
    model_sr: int = 48000,
    lyrics: str = "",
    edit_n_min: float = 0.5,
    edit_n_max: float = 1.0,
    edit_n_avg: int = 3,
    guidance_scale: float = 10.0,
    use_erg_tag: bool = True,
    infer_step: int = 50,
    layers_to_hook: str | list[str] | None = None,
    seed: int = 42,
    run_name: str | None = None,
):
    """
    Main function to edit audio files using ACE Step pipeline.

    Args:
        editing_type_id: Type of editing to perform (default: "0")
        dataset_name: Name of the dataset (default: "zome")
        model_sr: Model sample rate (default: 48000)
        lyrics: Lyrics for the audio (default: "")
        edit_n_min: Minimum edit parameter (default: 0.5)
        edit_n_max: Maximum edit parameter (default: 1.0)
        edit_n_avg: Average edit parameter (default: 3)
        guidance_scale: Guidance scale for generation (default: 10.0)
        use_erg_tag: Whether to use ERG tag (default: True)
        infer_step: Number of inference steps (default: 50)
        layers_to_hook: Layer(s) to hook - can be a string for single layer or list for multiple layers (default: None)
        seed: Random seed (default: 42)

    Returns:
        None
    """
    # Convert layers_to_hook to proper format
    if isinstance(layers_to_hook, str):
        if ',' in layers_to_hook:
            layers_to_hook = layers_to_hook.split(',')
        else:
            layers_to_hook = [layers_to_hook]
    elif layers_to_hook is None:
        # If None, use empty list (no hooking)
        layers_to_hook = []

    # Determine method name based on whether we're using hooks
    method_name = "flowedit"
    if layers_to_hook is not None and len(layers_to_hook) > 0:
        method_name = "ours_flowedit"

    # Create hyperparams dict from CLI arguments
    hyperparams = {
        "model_sr": model_sr,
        "lyrics": lyrics,
        "edit_n_min": edit_n_min,
        "edit_n_max": edit_n_max,
        "edit_n_avg": edit_n_avg,
        "guidance_scale": guidance_scale,
        "use_erg_tag": use_erg_tag,
        "infer_step": infer_step,
        "oss_steps": [],
        "seed": seed,
    }

    accelerator = Accelerator()
    ace_step = init_ace(accelerator.device)
    df_instruments = prepare_dataset(editing_type_id=editing_type_id, path_musiccaps=Path(PATH_MUSICCAPS))

    if run_name is None:
        dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        run_name = f"{method_name}_{dt_str}_nmin{hyperparams['edit_n_min']}_nmax{hyperparams['edit_n_max']}_navg{hyperparams['edit_n_avg']}_gs{hyperparams['guidance_scale']}_steps{hyperparams['infer_step']}"


    path_dir_outs = (
        PATH_DIR_OUTPUT
        / f"{dataset_name}_{editing_type_id}"
        / run_name
    ).resolve()
    path_dir_outs.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_instruments.iterrows(), total=len(df_instruments)):
        path_to_audio = row["path_yt"]
        source_prompt = row["original_prompt"]
        target_prompt = row["editing_prompt"]
        if layers_to_hook is not None and len(layers_to_hook) > 0:
            edited_audio, original_sr = edit_single_audio_hooked(
                path_to_audio=path_to_audio,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                ace_step_pipeline=ace_step,
                hyperparams=hyperparams,
                layers_to_hook=layers_to_hook,
            )
        else:
            edited_audio, original_sr = edit_single_audio(
                path_to_audio=path_to_audio,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                ace_step_pipeline=ace_step,
                hyperparams=hyperparams,
            )
        edited_audio = edited_audio.cpu()
        torchaudio.save((path_dir_outs / f"a{idx}.wav"), edited_audio, original_sr)


if __name__ == "__main__":
    fire.Fire(main)
