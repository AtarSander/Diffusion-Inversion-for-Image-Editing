from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torchaudio
from accelerate import Accelerator
from torchaudio.transforms import Resample
from tqdm import tqdm

from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
from editing.AudioEditingCode.code.env import PATH_AUDIOS_MEDLEY, PATH_PROMPTS_MEDLEY, PATH_EDIT_OUTPUTS


PATH_DIR_OUTPUT = Path(PATH_EDIT_OUTPUTS+"/medleymd").resolve()

MEDLEYMD_CLASSES = ["GENRE", "INSTR", "MOOD", "VOICE", "OTHER", "TEMPO"]
ACE_STEP_HOOKS = {
    "GENRE": [".ace_step_transformer.transformer_blocks.6", ".ace_step_transformer.transformer_blocks.7"],
    "INSTR": [".ace_step_transformer.transformer_blocks.5", ".ace_step_transformer.transformer_blocks.7"],
    "MOOD": [
        ".ace_step_transformer.transformer_blocks.6",
        ".ace_step_transformer.transformer_blocks.7",
        ".ace_step_transformer.transformer_blocks.13",
        ".ace_step_transformer.transformer_blocks.14",
    ],
    "VOICE": [
        ".ace_step_transformer.transformer_blocks.6",
        ".ace_step_transformer.transformer_blocks.7",
    ],
    "TEMPO": [
        ".ace_step_transformer.transformer_blocks.5",
        ".ace_step_transformer.transformer_blocks.6",
        ".ace_step_transformer.transformer_blocks.7",
    ],
    "OTHER": None,
}


def prepare_dataset(path_audios: Path, path_prompts: Path):
    df_prompts = pd.read_csv(path_prompts, index_col=0, header=0)
    df_instruments = []
    for idx, row in df_prompts.iterrows():
        filename = row["filename"]
        dirname = filename.split("_MIX")[0]
        path_to_audio = (path_audios / dirname / filename).resolve()
        df_instruments.append(
            {
                "path_yt": path_to_audio,
                "original_prompt": row["source_captions"],
                "editing_prompt": row["target_captions"],
                "edit_class": row["edit"],
            }
        )
    df_instruments = pd.DataFrame(df_instruments)
    return df_instruments


def init_ace(device):
    ace_step_pipeline = SimpleACEStepPipeline(device=device, dtype="bfloat16")
    ace_step_pipeline.load()
    return ace_step_pipeline


def edit_audio_ode_inversion(
    path_to_audio: Path,
    source_prompt: str,
    target_prompt: str,
    ace_step_pipeline: SimpleACEStepPipeline,
    hyperparams: dict,
    layers_to_hook: list[str] | None = None,
):
    original_audio, original_sr = torchaudio.load(path_to_audio)
    original_audio_len = original_audio.shape[1]
    if original_audio_len > 240 * original_sr:
        original_audio_len = 240 * original_sr
        print(f"WARNING: Audio {str(path_to_audio)} is too long and will be truncated to 240 seconds...")
    edited_audio = ace_step_pipeline.edit_audio_ode_inversion(
        src_audio_path=path_to_audio,
        source_prompt=source_prompt,
        source_lyrics=hyperparams["lyrics"],
        target_prompt=target_prompt,
        target_lyrics=hyperparams["lyrics"],
        tstart=hyperparams["tstart"],
        guidance_scale_inversion=hyperparams["guidance_scale_inversion"],
        guidance_scale_editing=hyperparams["guidance_scale_editing"],
        use_erg_tag=hyperparams["use_erg_tag"],
        infer_step=hyperparams["infer_step"],
        oss_steps=[],
        manual_seeds=hyperparams["seed"],
        cfg_type=hyperparams["cfg_type"],
        layers_to_hook=layers_to_hook,
    )
    resampler = Resample(hyperparams["model_sr"], original_sr)
    edited_audio_resampled = resampler(edited_audio)
    edited_audio_resampled = edited_audio_resampled[..., :original_audio_len]
    edited_audio_resampled = edited_audio_resampled.mean(dim=1)
    return edited_audio_resampled, original_sr


def main(
    model_sr: int = 48000,
    lyrics: str = "",
    use_erg_tag: bool = False,
    tstart: int = 50,
    guidance_scale_inversion: float = 1.0,
    guidance_scale_editing: float = 7.0,
    cfg_type: str = "cfg",
    infer_step: int = 50,
    with_hooks: bool = False,
    seed: int = 42,
    part_id: int = 0,
    n_parts: int | None = None,
    run_name: str | None = None,
):
    """
    Main function to edit audio files using ACE Step pipeline.

    Args:
        editing_type_id: Type of editing to perform (default: "0")
        dataset_name: Name of the dataset (default: "zome")
        model_sr: Model sample rate (default: 48000)
        lyrics: Lyrics for the audio (default: "")
        use_erg_tag: Whether to use ERG tag (default: True)
        tstart: Diffusion timestep to start the reverse process from (default: 50)
        guidance_scale_inversion: Guidance scale for inversion (default: 1.0)
        guidance_scale_editing: Guidance scale for editing (default: 7.0)
        cfg_type: Type of cfg (default: "cfg")
        infer_step: Number of inference steps (default: 50)
        with_hooks: Whether to use hooks (default: False)
        seed: Random seed (default: 42)

    Returns:
        None
    """
    # Determine method name based on whether we're using hooks
    method_name = "ode_inv"
    if with_hooks:
        method_name += "_ours"

    if n_parts is not None:
        assert 0 <= part_id < n_parts, f"part_id must be a number between 0 and {n_parts - 1} if n_parts is not None"

    # Create hyperparams dict from CLI arguments
    hyperparams = {
        "model_sr": model_sr,
        "lyrics": lyrics,
        "use_erg_tag": use_erg_tag,
        "tstart": tstart,
        "guidance_scale_inversion": guidance_scale_inversion,
        "guidance_scale_editing": guidance_scale_editing,
        "cfg_type": cfg_type,
        "infer_step": infer_step,
        "oss_steps": [],
        "seed": seed,
    }

    accelerator = Accelerator()
    ace_step = init_ace(accelerator.device)
    df_instruments = prepare_dataset(path_audios=Path(PATH_AUDIOS_MEDLEY), path_prompts=Path(PATH_PROMPTS_MEDLEY))

    if n_parts is None:
        dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        dt_str = datetime.now().strftime("%Y%m%d%H") + "0000" + "_parted"
    if run_name is None:
        run_name = f"{method_name}_{dt_str}_tstart{hyperparams['tstart']}_gsi{hyperparams['guidance_scale_inversion']}_gse{hyperparams['guidance_scale_editing']}_steps{hyperparams['infer_step']}"

    path_dir_outs = (PATH_DIR_OUTPUT / "ace" / run_name).resolve()
    path_dir_outs_audios = (path_dir_outs / "audios").resolve()
    path_dir_outs_audios.mkdir(parents=True, exist_ok=True)

    all_rows = list(df_instruments.iterrows())
    if n_parts is not None:
        ids = list(range(len(all_rows)))
        parts = np.array_split(ids, n_parts)
        parts = [list(p) for p in parts]
        all_rows = [all_rows[i] for i in parts[part_id]]

    for idx, row in tqdm(all_rows, total=len(all_rows)):
        path_to_audio = row["path_yt"]
        source_prompt = row["original_prompt"]
        target_prompt = row["editing_prompt"]
        edit_class = row["edit_class"]

        layers_to_hook = None
        if with_hooks:
            layers_to_hook = ACE_STEP_HOOKS[edit_class]
        edited_audio, original_sr = edit_audio_ode_inversion(
            path_to_audio=path_to_audio,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            ace_step_pipeline=ace_step,
            hyperparams=hyperparams,
            layers_to_hook=layers_to_hook,
        )
        edited_audio = edited_audio.cpu()
        torchaudio.save((path_dir_outs_audios / f"a{idx}.wav"), edited_audio, original_sr)


if __name__ == "__main__":
    fire.Fire(main)
