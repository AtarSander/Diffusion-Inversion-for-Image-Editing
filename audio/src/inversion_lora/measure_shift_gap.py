# ABOUTME: Measure the DDIM shifted-denoiser gap as a function of classifier-free guidance, to see
# ABOUTME: whether the error the inversion LoRA corrects grows with w before regenerating any data.

import json
import sys
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models import load_model  # noqa: E402


def combine(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, w: float) -> torch.Tensor:
    """Classifier-free guidance combination, the epsilon that actually advances the latent."""
    return eps_uncond + w * (eps_cond - eps_uncond)


@torch.no_grad()
def sample_gaps(
    ldm,
    sample_dir: Path,
    guidance_scale: float,
    batch_size: int,
    uncond: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Per-transition squared error of the shift substitution, for one trajectory.

    The shifted-denoiser approximation evaluates epsilon at the *cleaner* latent while passing the
    *noisier* timestep. The exact target is the epsilon that advanced the stored trajectory, which
    at w != 1 is the CFG combination rather than the conditional branch alone. Both are reported:
    `combined` is what a pair-branch loss would have to close, `cond_only` is what the current
    loss closes, and the difference isolates the combination from the path.

    Args:
        ldm: `AudioLDM2Wrapper`.
        sample_dir: One `sample_XXXXXX` directory holding both CFG branches.
        guidance_scale: The w the trajectory was generated with.
        batch_size: Transitions per forward pass.
        uncond: Unconditional `(hidden, t5, mask)`, shared across samples.

    Returns:
        `timestep`, `combined` and `cond_only` arrays, one entry per transition.
    """
    trajectory = torch.load(sample_dir / "latents/trajectory.pt", map_location="cpu")
    eps_cond_target = torch.load(sample_dir / "targets/target_eps.pt", map_location="cpu")
    uncond_path = sample_dir / "targets/uncond_eps.pt"
    assert uncond_path.exists(), (
        f"{sample_dir} has no uncond_eps.pt; regenerate with save_uncond_target=true, since at "
        "w != 1 the conditional branch alone does not describe the step that was taken"
    )
    eps_uncond_target = torch.load(uncond_path, map_location="cpu")
    timesteps = json.loads((sample_dir / "timesteps.json").read_text())
    cond = torch.load(sample_dir / "conditioning.pt", map_location="cpu")

    num = len(timesteps)
    assert trajectory.shape[0] == num + 1, (trajectory.shape, num)
    assert eps_cond_target.shape[0] == num == eps_uncond_target.shape[0]

    u_hidden, u_t5, u_mask = uncond
    hidden = cond["generated_prompt_embeds"][None].to(ldm.device)
    t5 = cond["t5_prompt_embeds"][None].to(ldm.device)
    mask = cond["t5_attention_mask"][None].to(ldm.device)

    combined, cond_only = [], []
    for start in range(0, num, batch_size):
        stop = min(start + batch_size, num)
        n = stop - start
        # The cleaner latent of transition i is trajectory[i + 1], carrying timestep[i].
        x_clean = trajectory[start + 1 : stop + 1].to(ldm.device, dtype=ldm.model.unet.dtype)
        t = torch.tensor(timesteps[start:stop], device=ldm.device)
        model_input = ldm.model.scheduler.scale_model_input(x_clean, t)

        eps_c = ldm.unet_forward(
            model_input,
            timestep=t,
            encoder_hidden_states=hidden.expand(n, -1, -1),
            class_labels=t5.expand(n, -1, -1),
            encoder_attention_mask=mask.expand(n, -1),
        )[0].sample
        eps_u = ldm.unet_forward(
            model_input,
            timestep=t,
            encoder_hidden_states=u_hidden.expand(n, -1, -1),
            class_labels=u_t5.expand(n, -1, -1),
            encoder_attention_mask=u_mask.expand(n, -1),
        )[0].sample

        target_c = eps_cond_target[start:stop].to(ldm.device, dtype=eps_c.dtype)
        target_u = eps_uncond_target[start:stop].to(ldm.device, dtype=eps_c.dtype)
        student = combine(eps_u, eps_c, guidance_scale)
        target = combine(target_u, target_c, guidance_scale)

        dims = tuple(range(1, student.ndim))
        combined.append(((student - target) ** 2).mean(dim=dims).float().cpu().numpy())
        cond_only.append(((eps_c - target_c) ** 2).mean(dim=dims).float().cpu().numpy())

    return {
        "timestep": np.array(timesteps),
        "combined": np.concatenate(combined),
        "cond_only": np.concatenate(cond_only),
    }


def band_report(gaps: dict[str, np.ndarray], num_bands: int = 4) -> dict[str, float]:
    """Mean gap overall and per equal band of the schedule, noisiest band first."""
    out = {
        "combined": float(gaps["combined"].mean()),
        "cond_only": float(gaps["cond_only"].mean()),
    }
    edges = np.linspace(0, 1000, num_bands + 1)
    for i in range(num_bands, 0, -1):
        low, high = edges[i - 1], edges[i]
        mask = (gaps["timestep"] > low) & (gaps["timestep"] <= high)
        name = f"q{int(high / 10):03d}_{int(low / 10):03d}"
        out[f"combined_{name}"] = float(gaps["combined"][mask].mean()) if mask.any() else float("nan")
    return out


def main(
    data_root: str,
    guidance_scale: float,
    device: str = "cuda:0",
    model_id: str = "cvssp/audioldm2-large",
    num_inference_steps: int = 200,
    batch_size: int = 8,
    limit_samples: int | None = None,
    out_root: str = "output/shift_gap",
) -> None:
    """Report the shift gap on a probe trajectory set generated at one guidance scale.

    Run once per guidance scale on paired sets (same prompts, same seed) and compare `combined`
    across them: that is the quantity a CFG-aware inversion LoRA would have to close, and whether
    it grows with w decides if regenerating the full dataset is worth it.

    Args:
        data_root: Probe dataset directory holding `sample_XXXXXX/`.
        guidance_scale: The w the set was generated with.
        device: CUDA device.
        model_id: AudioLDM2 checkpoint.
        num_inference_steps: DDIM grid the trajectories were generated on.
        batch_size: Transitions per forward pass.
        limit_samples: Score only the first N trajectories, for a smoke run.
        out_root: Parent for the timestamped output directory.
    """
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples = sorted(Path(data_root).glob("sample_*"))
    assert samples, f"no sample_* directories under {data_root}"
    if limit_samples is not None:
        samples = samples[:limit_samples]

    ldm = load_model(model_id, torch.device(device), num_inference_steps, edit_method="ddim")
    uncond = tuple(t.to(device) for t in ldm.encode_text([""], negative=True))

    per_sample, pooled = [], {"timestep": [], "combined": [], "cond_only": []}
    for i, sample_dir in enumerate(samples):
        gaps = sample_gaps(ldm, sample_dir, guidance_scale, batch_size, uncond)
        report = band_report(gaps)
        per_sample.append({"sample": sample_dir.name, **report})
        for key in pooled:
            pooled[key].append(gaps[key])
        logger.info(
            "{}/{} {}: combined={:.3e} cond_only={:.3e}",
            i + 1, len(samples), sample_dir.name, report["combined"], report["cond_only"],
        )

    overall = band_report({k: np.concatenate(v) for k, v in pooled.items()})
    logger.success("w={} over {} trajectories: {}", guidance_scale, len(samples), overall)

    out_dir = AUDIO_ROOT / out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}_shift_gap_g{guidance_scale}.json").write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "data_root": str(data_root),
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "num_trajectories": len(samples),
                "overall": overall,
                "per_sample": per_sample,
            },
            indent=2,
        )
    )
    logger.info("wrote {}", out_dir)


if __name__ == "__main__":
    fire.Fire(main)
