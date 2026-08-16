from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from audioldm_run import run_audioldm_edit
from tqdm import tqdm
import sys as _sys
_AUDIO_ROOT = Path(__file__).resolve().parents[3]
if str(_AUDIO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_AUDIO_ROOT))
from editing.dataset_medley import prepare_dataset  # noqa: E402

from run_paths import resolve_run_dir
from utils import set_reproducability

from env import PATH_AUDIOS_MEDLEY, PATH_PROMPTS_MEDLEY, PATH_EDIT_OUTPUTS

PATH_DIR_OUTPUT = Path(PATH_EDIT_OUTPUTS).resolve()

ENABLE_TQDM = False
AUDIOLDM_HOOKS = {
    "GENRE": [
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.1.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.2.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.5.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.5.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.6.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.6.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.9.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.9.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.1.attn2",
    ],
    "INSTR": [
        "up_blocks.1.attentions.5.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.5.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.9.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.9.transformer_blocks.1.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.1.attn2",
    ],
    "MOOD": [
        "up_blocks.1.attentions.5.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.0.attn2",
    ],
    "VOICE": [
        "up_blocks.1.attentions.5.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.10.transformer_blocks.0.attn2",
    ],
    "OTHER": None,
    "TEMPO": [
        "up_blocks.1.attentions.5.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.5.transformer_blocks.1.attn2",
    ],
}

dt_str = datetime.now().strftime("%Y%m%d%H%M%S")


def main(
    dataset_name: str = "medleymd",
    mode: str = "ddpm",
    num_diffusion_steps: int = 200,
    cfg_src: float = 3.0,
    cfg_tar: float = 12.0,
    tstart: int = 100,
    target_neg_prompt: str = "",
    seed: int = 42,
    run_name: str | None = None,
    with_hooks: bool = False,
    n_parts: int | None = None,
    part_id: int = 0,
    range_start: int | None = None,
    range_end: int | None = None,
    model_id: str = "cvssp/audioldm2-large",
    lora_path: str | None = None,
    unique_tracks: bool = False,
    reconstruct: bool = False,
):
    """
    Main function to edit audio files using AudioLDM2 methods.

    Args:
        dataset_name: Name of the dataset (default: "medleymd")
        mode: Editing mode - 'ddpm', 'ddim', or 'sdedit' (default: "ddpm")
        num_diffusion_steps: Number of diffusion steps (default: 200)
        cfg_src: Classifier-free guidance strength for forward process (default: 3.0)
        cfg_tar: Classifier-free guidance strength for reverse process (default: 12.0)
        tstart: Diffusion timestep to start the reverse process from (default: 100)
        target_neg_prompt: Negative prompt for target generation (default: "")
        seed: Random seed (default: 42)
        run_name: Custom run name (default: None - auto-generated)
        with_hooks: Whether to use cross-attention hooks for editing (default: False)
        n_parts: Number of parts to split dataset into for parallel processing (default: None)
        part_id: ID of the current part when using parallel processing (default: 0)
        model_id: AudioLDM model to use (default: "cvssp/audioldm2-large")
        lora_path: Trained inversion-LoRA checkpoint, applied to the DDIM inversion pass only
            (default: None)
        unique_tracks: Score one row per distinct MedleyDB track (35) instead of all 696
        reconstruct: Denoise with the *source* caption instead of the target one, turning the
            edit into a reconstruction of the input. Combined with cfg_tar=1.0 this measures how
            exactly inversion round-trips real audio (default: False)

    Returns:
        None
    """

    if n_parts is not None:
        assert (
            range_start is None and range_end is None
        ), "range_start and range_end must be None if n_parts is provided"
    assert n_parts is None or n_parts > 0, "n_parts must be a positive integer if provided"
    if n_parts is not None:
        assert 0 <= part_id < n_parts, f"part_id must be a number between 0 and {(n_parts-1)=} if n_parts is provided"

    assert mode in ["ddpm", "ddim", "sdedit"], "Invalid mode, must be one of ['ddpm', 'ddim', 'sdedit']"

    if range_start is not None:
        assert range_start >= 0, f"{range_start=} must be >=0"
        if range_end is not None:
            assert range_start <= range_end, f"{range_start=} must be <= {range_end=}"

    # Prepare dataset
    df_instruments = prepare_dataset(
        path_audios=Path(PATH_AUDIOS_MEDLEY),
        path_prompts=Path(PATH_PROMPTS_MEDLEY),
        unique_tracks=unique_tracks,
    )

    # Generate run name if not provided
    if run_name is None:
        dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        cfg_src_str = str(cfg_src)
        cfg_tar_str = str(cfg_tar)
        mode_name = mode
        if with_hooks:
            mode_name += "_ours"
        model_name = model_id.split("/")[-1]
        run_name = f"{model_name}_{mode_name}_{dt_str}_cfg_src{cfg_src_str}_cfg_tar{cfg_tar_str}_tstart{tstart}_steps{num_diffusion_steps}"

    # Create output directory
    path_dir_outs = resolve_run_dir(PATH_DIR_OUTPUT, dataset_name, f"audioldm2_{mode}", run_name)
    path_dir_outs.mkdir(parents=True, exist_ok=True)

    set_reproducability(seed, extreme=False)

    print(f"Processing {len(df_instruments)} audio files...")
    print(f"Mode: {mode}{' (reconstruction: denoising with the source caption)' if reconstruct else ''}")
    print(f"Output directory: {path_dir_outs}")

    all_rows = list(df_instruments.iterrows())
    if n_parts is not None:
        ids = list(range(len(all_rows)))
        parts = np.array_split(ids, n_parts)
        parts = [list(p) for p in parts]
        all_rows = [all_rows[i] for i in parts[part_id]]
    elif (range_start is not None) or (range_end is not None):
        range_start = range_start or 0
        range_end = len(all_rows) - 1 if range_end is None else range_end
        range_start = max(0, range_start)
        range_end = min(len(all_rows), range_end + 1)
        assert range_start < range_end, f"{range_start=} must be < {range_end=}"
        all_rows = all_rows[range_start:range_end]

    # Process each audio file in the dataset
    with tqdm(total=len(all_rows), desc="Processing files") as pbar:
        for idx, row in all_rows:
            path_to_audio = str(row["path_yt"])
            source_prompt = row["original_prompt"]
            target_prompt = row["original_prompt"] if reconstruct else row["editing_prompt"]
            edit_class = row["edit_class"]

            layers_to_hook = None
            if with_hooks:
                layers_to_hook = AUDIOLDM_HOOKS[edit_class]

            # Create output path for this specific file
            output_wav_path = str(path_dir_outs / f"a{idx}.wav")

            # Call the run_audioldm_edit function with save_edit_wav_path
            run_audioldm_edit(
                init_aud=path_to_audio,
                cfg_src=[cfg_src],
                cfg_tar=[cfg_tar],
                num_diffusion_steps=num_diffusion_steps,
                target_prompt=[target_prompt],
                source_prompt=[source_prompt],
                target_neg_prompt=[target_neg_prompt],
                tstart=[tstart],
                results_path="",  # Not used when save_edit_wav_path is provided
                mode=mode,
                verbose=False,  # Disable individual verbose to avoid spam
                save_edit_wav_path=output_wav_path,
                layers_to_hook=layers_to_hook,
                model_id=model_id,
                lora_path=lora_path,
            )
            pbar.update(1)

    print(f"\nCompleted processing {len(all_rows)} files.")
    print(f"Results saved to: {path_dir_outs}")


if __name__ == "__main__":
    fire.Fire(main)
