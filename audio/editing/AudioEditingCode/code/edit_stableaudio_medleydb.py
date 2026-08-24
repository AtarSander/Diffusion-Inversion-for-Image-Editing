from datetime import datetime
from pathlib import Path

import fire
from stable_audio_run import run_stable_audio_edit
from tqdm import tqdm
import numpy as np

from run_paths import resolve_run_dir
from utils import set_reproducability

import sys as _sys
from pathlib import Path as _Path
_AUDIO_ROOT = _Path(__file__).resolve().parents[3]
if str(_AUDIO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_AUDIO_ROOT))
from editing.dataset_medley import prepare_dataset  # noqa: E402

from env import PATH_AUDIOS_MEDLEY, PATH_EDIT_OUTPUTS, medley_split_paths

PATH_DIR_OUTPUT = Path(PATH_EDIT_OUTPUTS+"/medleymd").resolve()


STABLE_AUDIO_HOOKS = {
    "GENRE": [
        "transformer_blocks.11.attn2",
        "transformer_blocks.12.attn2",
        "transformer_blocks.13.attn2",
    ],  # ??
    "INSTR": [
        "transformer_blocks.11.attn2",
        "transformer_blocks.12.attn2",
    ],
    "MOOD": [
        "transformer_blocks.12.attn2",
        "transformer_blocks.13.attn2",
        "transformer_blocks.18.attn2",
    ],
    "VOICE": [
        "transformer_blocks.11.attn2",
    ],
    "OTHER": [
        "transformer_blocks.11.attn2",
        "transformer_blocks.12.attn2",
        "transformer_blocks.13.attn2",
        "transformer_blocks.18.attn2",
    ],
    "TEMPO": [
        "transformer_blocks.12.attn2",
    ],
}

dt_str = datetime.now().strftime("%Y%m%d%H%M%S")


def main(
    dataset_name: str = "medleymd",
    mode: str = "ddpm",
    num_diffusion_steps: int = 100,
    cfg_src: float = 1.0,
    cfg_tar: float = 3.5,
    tstart: int = 50,
    target_neg_prompt: str = "",
    seed: int = 42,
    run_name: str | None = None,
    with_hooks: bool = False,
    n_parts: int | None = None,
    part_id: int = 0,
    lora_path: str | None = None,
    unique_tracks: bool = False,
    reconstruct: bool = False,
    split: str = "full",
):
    """
    Main function to edit audio files using Stable Audio methods.

    Args:
        dataset_name: Name of the dataset (default: "medleymd")
        mode: Editing mode - 'ddpm', 'ddim', or 'sdedit' (default: "ddpm")
        num_diffusion_steps: Number of diffusion steps (default: 100)
        cfg_src: Classifier-free guidance strength for forward process (default: [1.0])
        cfg_tar: Classifier-free guidance strength for reverse process (default: [3.5])
        tstart: Diffusion timestep to start the reverse process from (default: 50)
        target_neg_prompt: Negative prompt for target generation (default: "")
        cutoff_points: Cutoff points for DDPM mode (default: None)
        fix_alpha: Alpha parameter for DDPM mode (default: 0.1)
        eta: Eta parameter for DDPM mode (default: 1.0)
        numerical_fix: Whether to apply numerical fix (default: True)
        seed: Random seed (default: 42)
        run_name: Custom run name (default: None - auto-generated)
        layers_to_hook: Layer(s) to hook for cross-attention replacement - can be a string for single layer or list for multiple layers (default: None)
        lora_path: Trained inversion-LoRA checkpoint, applied to the DDIM inversion pass only
            (default: None)
        unique_tracks: Score one row per distinct MedleyDB track (35) instead of every row
        reconstruct: Denoise with the *source* caption instead of the target one, turning the
            edit into a reconstruction of the input. Combined with cfg_tar=1.0 this measures how
            exactly inversion round-trips real audio (default: False)
        split: Benchmark split to edit, i.e. which prompt CSV to read (default: "full", all 696
            rows; "hparam" is the 115-row subset used for the sweeps). Outputs are named by the
            row's index in the full set whichever split is used, so the eval has to be given the
            same split.

    Returns:
        None
    """

    assert n_parts is None or n_parts > 0, "n_parts must be a positive integer if provided"
    if n_parts is not None:
        assert 0 <= part_id < n_parts, f"part_id must be a number between 0 and {(n_parts-1)=} if n_parts is provided"

    assert mode in ["ddpm", "ddim", "sdedit", "odeinv"], (
        "Invalid mode, must be one of ['ddpm', 'ddim', 'sdedit', 'odeinv']"
    )

    # Prepare dataset
    prompts_csv, _ = medley_split_paths(split)
    df_instruments = prepare_dataset(
        path_audios=Path(PATH_AUDIOS_MEDLEY),
        path_prompts=Path(prompts_csv),
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
        run_name = f"stable_audio_{mode_name}_{dt_str}_cfg_src{cfg_src_str}_cfg_tar{cfg_tar_str}_tstart{tstart}_steps{num_diffusion_steps}"

    # Create output directory
    path_dir_outs = resolve_run_dir(PATH_DIR_OUTPUT, dataset_name, "stable_audio", run_name)
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

    # Process each audio file in the dataset
    with tqdm(total=len(all_rows), desc="Processing files") as pbar:
        for idx, row in all_rows:
            path_to_audio = str(row["path_yt"])
            source_prompt = row["original_prompt"]
            target_prompt = row["original_prompt"] if reconstruct else row["editing_prompt"]
            edit_class = row["edit_class"]

            layers_to_hook = None
            if with_hooks:
                layers_to_hook = STABLE_AUDIO_HOOKS[edit_class]

            # Create output path for this specific file
            output_wav_path = str(path_dir_outs / f"a{idx}.wav")

            # Call the original run_stable_audio_edit function with save_edit_wav_path
            run_stable_audio_edit(
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
                lora_path=lora_path,
            )
            pbar.update(1)

    print(f"\nCompleted processing {len(all_rows)} files.")
    print(f"Results saved to: {path_dir_outs}")


if __name__ == "__main__":
    fire.Fire(main)
