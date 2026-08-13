# ABOUTME: Profile the per-step DDIM inversion approximation error across the noise schedule, the
# ABOUTME: exact quantity the inversion LoRA is trained to remove, from a cached trajectory set.

import json
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
for _path in (AUDIO_ROOT, AUDIO_ROOT / "editing/AudioEditingCode/code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models import load_model  # noqa: E402

NUM_QUARTILES = 4


@torch.no_grad()
def gap_for_sample(ldm, sample_dir: Path, timesteps: np.ndarray, batch_size: int):
    """Per-transition inversion error, signed mean and teacher magnitude for one trajectory.

    DDIM inversion needs eps(x_t, t) to reach x_t from x_{t-1}, but x_t is what it is solving
    for, so it substitutes eps(x_{t-1}, t). This measures that substitution error directly.

    Args:
        ldm: `AudioLDM2Wrapper` with the frozen teacher.
        sample_dir: One `sample_*` directory.
        timesteps: The DDIM grid, noisiest first.
        batch_size: Transitions evaluated per forward pass.

    Returns:
        Arrays of RMS error, signed mean error and teacher RMS, one entry per transition.
    """
    target = torch.load(sample_dir / "targets/target_eps.pt", map_location="cpu")
    trajectory = torch.load(sample_dir / "latents/trajectory.pt", map_location="cpu")
    conditioning = torch.load(sample_dir / "conditioning.pt", map_location="cpu")
    assert target.shape[0] == len(timesteps), (target.shape, len(timesteps))

    device = ldm.device
    hidden = conditioning["generated_prompt_embeds"].unsqueeze(0).to(device)
    t5 = conditioning["t5_prompt_embeds"].unsqueeze(0).to(device)
    mask = conditioning["t5_attention_mask"].unsqueeze(0).to(device)

    rms, signed, magnitude = [], [], []
    for start in range(0, len(timesteps), batch_size):
        stop = min(start + batch_size, len(timesteps))
        count = stop - start
        # The student's input: the cleaner latent, paired with the noisier timestep.
        x_clean = trajectory[start + 1 : stop + 1].to(device)
        t = torch.tensor(timesteps[start:stop], dtype=torch.long, device=device)
        model_input = ldm.model.scheduler.scale_model_input(x_clean, t[0])
        shifted = ldm.unet_forward(
            model_input,
            timestep=t,
            encoder_hidden_states=hidden.expand(count, -1, -1),
            class_labels=t5.expand(count, -1, -1),
            encoder_attention_mask=mask.expand(count, -1),
        )[0].sample.cpu()

        gap = shifted - target[start:stop]
        rms.append(gap.flatten(1).pow(2).mean(dim=1).sqrt().numpy())
        signed.append(gap.flatten(1).mean(dim=1).numpy())
        magnitude.append(target[start:stop].flatten(1).pow(2).mean(dim=1).sqrt().numpy())

    return np.concatenate(rms), np.concatenate(signed), np.concatenate(magnitude)


def main(
    root_dir: str,
    device: str = "cuda:0",
    num_samples: int | None = None,
    batch_size: int = 8,
    num_inference_steps: int = 200,
    output_json: str | None = None,
) -> None:
    """Profile the inversion error by noise-schedule quartile over a trajectory set.

    Args:
        root_dir: Cached trajectory dataset.
        device: Torch device.
        num_samples: Trajectories to use; None uses all.
        batch_size: Transitions per forward pass.
        num_inference_steps: DDIM grid length; must match the dataset.
        output_json: Optional path to write the per-step profile to.
    """
    samples = sorted(Path(root_dir).glob("sample_*"))[:num_samples]
    assert samples, f"no sample_* directories under {root_dir}"
    timesteps = np.array(json.loads((samples[0] / "timesteps.json").read_text()))

    ldm = load_model("cvssp/audioldm2-large", device, num_inference_steps, edit_method="ddim")
    logger.info("Profiling {} trajectories x {} transitions", len(samples), len(timesteps))

    rms, signed, magnitude = [], [], []
    for index, sample_dir in enumerate(samples):
        sample_rms, sample_signed, sample_magnitude = gap_for_sample(
            ldm, sample_dir, timesteps, batch_size
        )
        rms.append(sample_rms)
        signed.append(sample_signed)
        magnitude.append(sample_magnitude)
        if index == 0:
            logger.info(
                "First trajectory: gap RMS {:.5f} at t={}, {:.5f} at t={}",
                sample_rms[0], timesteps[0], sample_rms[-1], timesteps[-1],
            )
        logger.info("{}/{} {}", index + 1, len(samples), sample_dir.name)

    rms = np.stack(rms)
    signed = np.stack(signed)
    magnitude = np.stack(magnitude)

    fraction = 1.0 - timesteps / 1000.0
    quartile = np.clip((fraction * NUM_QUARTILES).astype(int), 0, NUM_QUARTILES - 1)

    print(f"\n{len(samples)} trajectories, {len(timesteps)} steps, no CFG\n")
    header = f"{'quartile':>13} {'t range':>12} {'sigma_eps':>10} {'e_RMS':>9} {'+/- sd':>9} {'e_MSE':>11} {'e_rel':>8} {'bias/e_RMS':>11}"
    print(header)
    rows = {}
    for q in range(NUM_QUARTILES):
        mask = quartile == q
        label = "noisiest" if q == 0 else ("cleanest" if q == 3 else "")
        e_rms = rms[:, mask].mean()
        # Spread across trajectories of that quartile's mean error.
        sd = rms[:, mask].mean(axis=1).std()
        sigma = magnitude[:, mask].mean()
        bias = np.abs(signed[:, mask]).mean()
        rows[f"q{q + 1}"] = {
            "t_max": int(timesteps[mask].max()), "t_min": int(timesteps[mask].min()),
            "sigma_eps": float(sigma), "e_RMS": float(e_rms), "e_RMS_sd": float(sd),
            "e_MSE": float((rms[:, mask] ** 2).mean()), "e_rel": float(e_rms / sigma),
            "bias_ratio": float(bias / e_rms),
        }
        print(
            f"  q{q + 1} {label:>9} {timesteps[mask].max():>5}..{timesteps[mask].min():<5} "
            f"{sigma:>10.4f} {e_rms:>9.5f} {sd:>9.5f} {(rms[:, mask] ** 2).mean():>11.3e} "
            f"{e_rms / sigma:>7.2%} {bias / e_rms:>11.4f}"
        )

    print(
        f"\nq4/q1 ratios: e_RMS {rows['q4']['e_RMS'] / rows['q1']['e_RMS']:.0f}x   "
        f"e_MSE {rows['q4']['e_MSE'] / rows['q1']['e_MSE']:.0f}x   "
        f"e_rel {rows['q4']['e_rel'] / rows['q1']['e_rel']:.0f}x"
    )
    print(
        "bias/e_RMS is the signed mean error over its RMS: ~0 means the error is symmetric "
        "noise with no constant offset for the adapter to absorb."
    )

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_trajectories": len(samples),
                    "timesteps": timesteps.tolist(),
                    "quartiles": rows,
                    "per_step_e_RMS": rms.mean(axis=0).tolist(),
                    "per_step_sigma_eps": magnitude.mean(axis=0).tolist(),
                },
                f,
                indent=2,
            )
        logger.success("Wrote {}", output_json)


if __name__ == "__main__":
    fire.Fire(main)
