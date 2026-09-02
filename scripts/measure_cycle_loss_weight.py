# ABOUTME: Measures the real weight of the cycle loss relative to L_inv on SD1.4 DDIM at cfg=1,
# ABOUTME: by running actual trajectories and evaluating eps at z_i, z_{i+1} and the inverted z_hat.

import json
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def sqn(x: torch.Tensor) -> float:
    """Mean squared magnitude, i.e. the MSE-style loss value."""
    return float(x.pow(2).mean())


@torch.no_grad()
def run_prompt(pipe, sched, prompt: str, seed: int, device, dtype):
    """Sample one DDIM trajectory and measure L_inv / L_cycle at every transition.

    Convention follows the loss formula: z_i is the NOISIER point (level t_i), z_{i+1} the
    cleaner one. Generation is z_i -> z_{i+1}. The adapter is at LoRA init, where it is exactly
    the identity, so eps_phi(z_{i+1}, t_i) == eps_theta(z_{i+1}, t_i).

    Returns:
        A list of per-transition dicts.
    """
    embed = pipe.encode_prompt(prompt, device, 1, do_classifier_free_guidance=False)[0]
    unet, T = pipe.unet, sched.config.num_train_timesteps
    step_ratio = T // len(sched.timesteps)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float64)

    def eps(z, t):
        return unet(z.to(dtype), t, encoder_hidden_states=embed).sample.to(torch.float32)

    def coeffs(t):
        """Affine DDIM coefficients A, B for the generation step starting at timestep t."""
        prev = int(t) - step_ratio
        a_noisy = abar[int(t)]
        a_clean = abar[prev] if prev >= 0 else sched.final_alpha_cumprod.to(torch.float64)
        A = (a_clean / a_noisy).sqrt()
        B = (1 - a_clean).sqrt() - A * (1 - a_noisy).sqrt()
        return float(A), float(B)

    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(1, unet.config.in_channels, 64, 64, generator=g).to(device) * sched.init_noise_sigma

    # Forward sampling pass: keep every latent and the eps each step consumed.
    lat, used = [z], []
    for t in sched.timesteps:
        u = eps(z, t)
        A, B = coeffs(t)
        z_next = A * z + B * u
        # Validate the affine form against diffusers' own step().
        ref = sched.step(u.to(dtype), t, z.to(dtype), eta=0.0).prev_sample.to(torch.float32)
        err = float((z_next - ref).abs().max())
        assert err < 2e-3, f"affine A,B disagrees with scheduler.step at t={int(t)}: {err:.2e}"
        used.append(u)
        lat.append(z_next)
        z = z_next

    rows = []
    for k, t in enumerate(sched.timesteps):
        A, B = coeffs(t)
        z_i, z_ip1, u = lat[k], lat[k + 1], used[k]
        e = eps(z_ip1, t)                      # shifted call: cleaner latent, noisier timestep
        z_hat = (z_ip1 - B * e) / A
        v = eps(z_hat, t)

        l_inv = sqn(e - u)
        l_cycle = B**2 * sqn(e - v)
        # Identity check: z_hat - z_i must equal -(B/A)(e-u) exactly.
        ident = float((z_hat - z_i + (B / A) * (e - u)).abs().max()) / max(
            float((z_hat - z_i).abs().max()), 1e-12
        )
        dz = float((z_hat - z_i).norm())
        rows.append({
            "t": int(t), "A": A, "B": B, "B2": B**2,
            "L_inv": l_inv, "L_cycle": l_cycle, "ratio": l_cycle / l_inv,
            "resid_ratio": (sqn(e - v) / l_inv) ** 0.5,          # ||e-v|| / ||e-u||
            "J_eff": float((v - u).norm()) / dz if dz > 0 else 0.0,
            "identity_rel_err": ident,
            "latent_inv_err": dz, "z_i_norm": float(z_i.norm()),
        })
    return rows


def main(
    device: str = "cuda:0",
    model_key: str = "CompVis/stable-diffusion-v1-4",
    num_ddim_steps: int = 50,
    prompts: tuple = (
        "a photograph of a cat sitting on a wooden bench",
        "an oil painting of a mountain village at sunset",
        "a bowl of fresh strawberries on a marble counter",
    ),
    seed: int = 42,
    output_dir: str = "output/cycle_loss_weight",
) -> None:
    """Measure L_cycle/L_inv on SD1.4 DDIM at cfg=1 with lambda_cycle=1.0.

    Args:
        device: Torch device, set explicitly.
        model_key: Hub id.
        num_ddim_steps: Length of the DDIM grid (50 matches pnp_inversion).
        prompts: Captions to average over.
        seed: Seed for the initial latent.
        output_dir: Destination, relative to the repo root.
    """
    load_dotenv(ROOT / ".env", override=True)
    dev, dtype = torch.device(device), torch.float32
    torch.cuda.set_device(dev)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_key, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(dev)
    raw_cfg = dict(pipe.scheduler.config)
    sched = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    print(f"scheduler config as shipped: clip_sample={raw_cfg.get('clip_sample')} "
          f"set_alpha_to_one={raw_cfg.get('set_alpha_to_one')} "
          f"steps_offset={raw_cfg.get('steps_offset')} "
          f"beta_schedule={raw_cfg.get('beta_schedule')}")
    sched.register_to_config(clip_sample=False)  # clipping x0 makes the step non-affine
    sched.set_timesteps(num_ddim_steps, device=dev)
    print(f"grid: {sched.timesteps[:3].tolist()} ... {sched.timesteps[-3:].tolist()}")

    all_rows = []
    for p in prompts:
        rows = run_prompt(pipe, sched, p, seed, dev, dtype)
        for r in rows:
            r["prompt"] = p
        all_rows += rows
        logger.info("{}: mean ratio {:.4e}", p[:40], sum(r["ratio"] for r in rows) / len(rows))

    n = len(all_rows)
    mean = lambda k: sum(r[k] for r in all_rows) / n  # noqa: E731
    print("\n" + "=" * 78)
    print(f"MEASURED on {model_key} DDIM-{num_ddim_steps}, cfg=1, lambda_cycle=1.0, "
          f"LoRA at init, {n} transitions over {len(prompts)} prompts")
    print("=" * 78)
    print(f"max identity relative error : {max(r['identity_rel_err'] for r in all_rows):.3e}"
          "   (z_hat - z_i == -(B/A)(e-u))")
    print(f"mean ||e-v|| / ||e-u||      : {mean('resid_ratio'):.4f}")
    print(f"mean effective ||J||        : {mean('J_eff'):.3f}")
    print(f"mean contraction |B/A|*||J||: "
          f"{sum(abs(r['B'] / r['A']) * r['J_eff'] for r in all_rows) / n:.4f}")
    print(f"\nmean L_inv                  : {mean('L_inv'):.4e}")
    print(f"mean L_cycle                : {mean('L_cycle'):.4e}")
    print(f"mean B^2                    : {mean('B2'):.4e}")
    ratios = sorted(r["ratio"] for r in all_rows)
    print(f"\nL_cycle / L_inv  mean {mean('ratio'):.4e}  ({100 * mean('ratio'):.3f}%)")
    print(f"                 min  {ratios[0]:.4e}  max {ratios[-1]:.4e}")
    print(f"                 -> 1 part in {1 / mean('ratio'):.0f}")
    print(f"lambda_cycle for parity     : {1 / mean('ratio'):.0f}")

    print(f"\n{'t':>5} {'B^2':>10} {'L_inv':>11} {'L_cycle':>11} {'ratio':>10} "
          f"{'|e-v|/|e-u|':>12} {'J_eff':>7}")
    first = [r for r in all_rows if r["prompt"] == prompts[0]]
    for r in first[::6] + [first[-1]]:
        print(f"{r['t']:>5} {r['B2']:>10.3e} {r['L_inv']:>11.4e} {r['L_cycle']:>11.4e} "
              f"{r['ratio']:>10.3e} {r['resid_ratio']:>12.4f} {r['J_eff']:>7.3f}")

    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"cycle_weight_sd14_ddim{num_ddim_steps}_{stamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "model_key": model_key, "num_ddim_steps": num_ddim_steps, "seed": seed,
            "cfg": 1.0, "dtype": "float32", "scheduler_config_as_shipped": raw_cfg,
            "summary": {
                "mean_ratio": mean("ratio"), "min_ratio": ratios[0], "max_ratio": ratios[-1],
                "mean_B2": mean("B2"), "mean_L_inv": mean("L_inv"),
                "mean_L_cycle": mean("L_cycle"), "mean_resid_ratio": mean("resid_ratio"),
                "mean_J_eff": mean("J_eff"),
                "max_identity_rel_err": max(r["identity_rel_err"] for r in all_rows),
            },
            "rows": all_rows,
        }, f, indent=2)
    logger.success("Wrote {}", path)


if __name__ == "__main__":
    fire.Fire(main)
