# ABOUTME: Tests whether the real-audio (forward-noise) Stable Audio pairs' 370x LoRA-off gap comes from
# ABOUTME: a duration-conditioning mismatch: targets at the clip's duration, trainer queries at the max.

import sys
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))


@torch.no_grad()
def lora_off_gap(teacher, solver, rows, targets, timesteps, text_audio, batch_size: int):
    """Per-pair MSE of the frozen teacher at each pair's input against its target.

    Exactly the trainer's LoRA-disabled loss on a real-pairs sample: the input `rows[k + 1]` at its
    own (matched) timestep, the stored cross-attention states, and whatever global timing the
    teacher currently holds -- the one thing this check varies.
    """
    device = teacher.device
    inputs = rows[1:].to(device)
    ts = torch.tensor(timesteps, device=device, dtype=torch.float32)
    sigmas = solver.sigma_for(ts).to(device=device, dtype=inputs.dtype)
    out = []
    for lo in range(0, inputs.shape[0], batch_size):
        hi = min(lo + batch_size, inputs.shape[0])
        raw = teacher.forward(solver.model_input_batch(inputs[lo:hi], sigmas[lo:hi]), ts[lo:hi],
                              text_audio)
        pred = solver.data_prediction_batch(inputs[lo:hi], raw, sigmas[lo:hi])
        out.append((pred.float() - targets[lo:hi].to(device).float()).pow(2).flatten(1).mean(1))
    return torch.cat(out).cpu()


def main(num_clips: int = 16, steps: int = 100, device: str = "cuda:0", batch_size: int = 8,
         num_captions: int = 1500, seed_base: int = 500000):
    """Rebuild real-audio pairs with the generator's own code and score the LoRA-off gap twice.

    Args:
        num_clips: MusicCaps clips to use (the first ones on disk among the training captions).
        steps: Grid length of the real-pairs dataset (100, the saocos grid).
        device: Torch device.
        batch_size: DiT forward chunk size.
        num_captions: Leading caption rows the real-pairs dataset drew from.
        seed_base: Seed base of the real-pairs dataset (its first draw per clip).
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    from src.inversion_lora.generate_real_pairs_stable_audio import (
        encode_clip,
        load_clip_records,
        real_pairs,
    )
    from src.inversion_lora.stable_audio import ExactDPMSolver, load_teacher

    records = load_clip_records(
        AUDIO_ROOT / "editing/music_caps_dl/metadata/musiccaps-public.csv",
        Path("/nas/lstanisz/data/musiccaps/audio"), num_captions,
    )[:num_clips]
    assert records, "no MusicCaps clips found"
    dev = torch.device(device)
    teacher = load_teacher("stabilityai/stable-audio-open-1.0", dev, steps, schedule="cosine")
    solver = ExactDPMSolver(teacher.model.scheduler)
    max_s = teacher.max_duration_s

    gaps = {"mismatched": [], "matched": []}
    durations = []
    for i, record in enumerate(records):
        # The generator: encode at the max window, then set the clip's duration for the prompt
        # embedding and every teacher target.
        teacher.set_duration(max_s)
        x0, clip_s = encode_clip(teacher, record["wav"])
        teacher.set_duration(clip_s)
        text_audio = teacher.encode_prompt(record["prompt"])
        rows, targets, _, timesteps = real_pairs(teacher, solver, x0, text_audio, seed_base + i * 8,
                                                 batch_size)
        # The trainer loads its teacher once at duration_s=null (the max) and never changes it.
        teacher.set_duration(max_s)
        gaps["mismatched"].append(lora_off_gap(teacher, solver, rows, targets, timesteps,
                                               text_audio, batch_size))
        teacher.set_duration(clip_s)
        gaps["matched"].append(lora_off_gap(teacher, solver, rows, targets, timesteps, text_audio,
                                            batch_size))
        durations.append(clip_s)
        logger.info("clip {}/{} ({:.1f}s): mismatched {:.3e}  matched {:.3e}", i + 1, len(records),
                    clip_s, float(gaps["mismatched"][-1].mean()), float(gaps["matched"][-1].mean()))

    n_pairs = len(gaps["matched"][0])
    bands = [(0, n_pairs // 4), (n_pairs // 4, n_pairs // 2), (n_pairs // 2, 3 * n_pairs // 4),
             (3 * n_pairs // 4, n_pairs)]
    print(f"\n{len(records)} MusicCaps clips (durations {min(durations):.1f}-{max(durations):.1f}s), "
          f"{steps}-step cosine grid, {n_pairs} pairs each; LoRA-off MSE, data-prediction units")
    print(f"{'condition':<12}{'pooled':>12}" + "".join(f"{f'pairs {a}-{b - 1}':>16}" for a, b in bands))
    for name, values in gaps.items():
        stacked = torch.stack(values)  # [clips, pairs]
        cells = "".join(f"{float(stacked[:, a:b].mean()):>16.3e}" for a, b in bands)
        print(f"{name:<12}{float(stacked.mean()):>12.3e}{cells}")
    ratio = float(torch.stack(gaps["mismatched"]).mean() / torch.stack(gaps["matched"]).mean())
    print(f"mismatched / matched = {ratio:.1f}x   (reference: the realfn dataset's logged LoRA-off "
          f"loss 9.75e-2 vs 2.63e-4 on trajectories)")


if __name__ == "__main__":
    fire.Fire(main)
