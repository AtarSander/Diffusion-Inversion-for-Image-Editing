# ABOUTME: Measure the AudioLDM2 UNet forward cost that dominates every edit, under the precision
# ABOUTME: and batching settings we could change, reporting speedup next to the epsilon error.

import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import fire
import torch

AUDIO_ROOT = Path(__file__).resolve().parents[1]
for path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import load_model  # noqa: E402
from utils import set_reproducability  # noqa: E402

from src.inversion_lora.generate_trajectories import latent_height  # noqa: E402


def timed_forwards(call, repeats: int) -> tuple[float, torch.Tensor]:
    """Median wall-clock per forward, and the last epsilon, with the GPU synchronised.

    Args:
        call: Zero-argument callable running one UNet forward.
        repeats: Timed iterations after one warm-up.

    Returns:
        `(median_seconds, last_epsilon)`.
    """
    call()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        eps = call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    times.sort()
    return times[len(times) // 2], eps


def main(
    device: str = "cuda:0",
    duration_s: float = 60.0,
    model_id: str = "cvssp/audioldm2-large",
    repeats: int = 8,
    prompt: str = "A recording of an old upbeat cool jazz song.",
    out_root: str = "output/unet_profile",
):
    """Profile one UNet forward at the duration the edit pipeline actually uses.

    Every edit is 200-800 of these forwards, so this is where the wall-clock of the whole
    benchmark is decided. Each variant reports its speedup over the settings the drivers run
    today plus the relative deviation of its epsilon, since anything that changes epsilon changes
    the inversion the edits depend on.

    Args:
        device: CUDA device to profile on.
        duration_s: Input length in seconds; the drivers truncate to 60.
        model_id: AudioLDM2 checkpoint.
        repeats: Timed forwards per variant.
        prompt: Conditioning text.
        out_root: Directory for the timestamped JSON result.
    """
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    # Exactly what the edit drivers do before touching the model.
    set_reproducability(42, extreme=False)

    ldm = load_model(model_id, torch_device, 200, edit_method="ddim")
    hidden, class_labels, mask = ldm.encode_text([prompt])

    # The drivers derive this from the input audio; latent_height mirrors the pipeline exactly.
    height = latent_height(ldm.model, duration_s)
    latent = torch.randn(
        1,
        8,
        height // ldm.model.vae_scale_factor,
        16,
        device=torch_device,
        dtype=torch.float32,
    )
    timestep = torch.tensor([501], device=torch_device)
    print(f"latent {tuple(latent.shape)}  t5 {tuple(hidden.shape)}  duration {duration_s}s")

    def forward(x, autocast_dtype=None):
        context = (
            torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext()
        )
        with torch.inference_mode(), context:
            return ldm.unet_forward(
                x,
                timestep=timestep,
                encoder_hidden_states=hidden.repeat(x.shape[0], 1, 1),
                class_labels=class_labels.repeat(x.shape[0], 1, 1),
                encoder_attention_mask=mask.repeat(x.shape[0], 1),
            )[0].sample

    def settings(matmul_tf32: bool, cudnn_tf32: bool, benchmark: bool):
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark

    # name -> (matmul tf32, cudnn tf32, cudnn benchmark, autocast dtype, batch)
    VARIANTS = {
        "as-shipped": (True, False, False, None, 1),
        "cudnn tf32": (True, True, False, None, 1),
        "cudnn tf32 + benchmark": (True, True, True, None, 1),
        "no tf32 at all": (False, False, False, None, 1),
        "bf16 autocast": (True, True, False, torch.bfloat16, 1),
        "fp16 autocast": (True, True, False, torch.float16, 1),
        "cudnn tf32, CFG batched": (True, True, False, None, 2),
    }

    reference = None
    baseline = None
    rows = []
    for name, (matmul_tf32, cudnn_tf32, benchmark, dtype, batch) in VARIANTS.items():
        settings(matmul_tf32, cudnn_tf32, benchmark)
        x = latent.repeat(batch, 1, 1, 1)
        seconds, eps = timed_forwards(lambda: forward(x, dtype), repeats)
        # A batched pair replaces two sequential forwards, so its comparable unit is per sample.
        per_step = seconds / batch
        if reference is None:
            reference, baseline, error = eps.float(), per_step, 0.0
        else:
            error = float((eps[:1].float() - reference).abs().max() / reference.abs().max())
        rows.append(
            {
                "variant": name,
                "seconds_per_forward": per_step,
                "speedup": baseline / per_step,
                "rel_eps_error": error,
                "batch": batch,
            }
        )
        print(
            f"{name:26s} {per_step * 1000:7.1f} ms/forward  {baseline / per_step:5.2f}x  "
            f"eps rel. err {error:.2e}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "device": torch.cuda.get_device_name(torch_device),
        "model_id": model_id,
        "duration_s": duration_s,
        "latent_shape": list(latent.shape),
        "repeats": repeats,
        "variants": rows,
    }
    (out_dir / f"{stamp}_unet_profile.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / f'{stamp}_unet_profile.json'}")

    edits = {"sdedit t100": 200, "ddpm t100": 600, "ddim t200": 800}
    print("\nprojected per-edit cost (evals x ms/forward + 2 s fixed):")
    for row in rows:
        line = "  ".join(
            f"{name} {evals * row['seconds_per_forward'] + 2:6.1f} s" for name, evals in edits.items()
        )
        print(f"  {row['variant']:26s} {line}")


if __name__ == "__main__":
    fire.Fire(main)
