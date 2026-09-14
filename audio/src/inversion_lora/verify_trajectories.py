# ABOUTME: Validate a cached trajectory dataset (AudioLDM2 DDIM or Stable Audio ODE): no partial
# ABOUTME: samples, consistent shapes, and stepping with the cached target lands on the next row.

import json
import random
import sys
from pathlib import Path

import fire
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

AUDIO_ROOT = Path(__file__).resolve().parents[2]
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

STEP_TOLERANCE = 1e-4
# Relative, because Stable Audio latents span sigma 500..0.3: recomputing the fp32 step of
# fp32-stored tensors moves a few ulp, never 1e-4 of the norm.
SAO_STEP_TOLERANCE = 1e-4


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def check_sao_step(
    sample_dir: Path,
    meta: dict,
    traj: torch.Tensor,
    eps: torch.Tensor,
    timesteps: list,
    scheduler_cache: dict,
) -> list[str]:
    """Recompute the exact first-order ODE step invariant for a Stable Audio sample.

    A plain dataset satisfies `forward(traj[i], eps[i], i) == traj[i + 1]` because that is how
    the rows were produced. A dense dataset (`input_space == "exact_coarse_step"`) anchors each
    step at the fine states saved in `latents/states.pt` instead of at the previous row.

    Args:
        sample_dir: Directory holding the sample.
        meta: The sample's meta.json contents.
        traj: Saved rows `[N + 1, C, L]`.
        eps: Saved data-prediction targets `[N, C, L]`.
        timesteps: Cached grid, one entry per pair.
        scheduler_cache: Reused `ExactDPMSolver` instances keyed by (model_id, num_steps).

    Returns:
        Human-readable problem descriptions (empty when healthy).
    """
    from diffusers import CosineDPMSolverMultistepScheduler

    from src.inversion_lora.stable_audio import ExactDPMSolver

    key = (meta["model_id"], meta["num_inference_steps"])
    if key not in scheduler_cache:
        sched = CosineDPMSolverMultistepScheduler.from_pretrained(
            meta["model_id"], subfolder="scheduler"
        )
        sched.set_timesteps(meta["num_inference_steps"], device="cpu")
        scheduler_cache[key] = ExactDPMSolver(sched)
    solver = scheduler_cache[key]

    grid = torch.tensor(solver.timesteps[1:], dtype=torch.float64)
    cached = torch.tensor(timesteps, dtype=torch.float64)
    if cached.shape != grid.shape or not torch.allclose(cached, grid, atol=1e-5):
        return [f"{sample_dir.name}: cached timesteps do not match the solver grid"]

    if meta.get("input_space") == "exact_coarse_step":
        states_path = sample_dir / "latents/states.pt"
        if not states_path.exists():
            return [f"{sample_dir.name}: dense sample without latents/states.pt"]
        base = _load(states_path)[:-1]
        if base.shape != traj[:-1].shape:
            return [f"{sample_dir.name}: states shape {tuple(base.shape)} != trajectory"]
    else:
        base = traj[:-1]

    a, b = solver.coefficients_batch(torch.arange(eps.shape[0]))
    got = a * base.float() + b * eps.float()
    want = traj[1:].float()
    rel = (got - want).flatten(1).norm(dim=1) / want.flatten(1).norm(dim=1)
    if float(rel.max()) > SAO_STEP_TOLERANCE:
        return [
            f"{sample_dir.name}: ODE step invariant violated, worst rel={float(rel.max()):.3e}"
        ]
    return []


def check_sample(sample_dir: Path, scheduler_cache: dict, check_step: bool) -> list[str]:
    """Check one sample directory, returning a list of problems (empty when healthy).

    Args:
        sample_dir: Directory holding one cached trajectory.
        scheduler_cache: Reused DDIMScheduler instances keyed by (model_id, num_steps).
        check_step: Whether to re-run the DDIM step invariant (needs the scheduler config).

    Returns:
        Human-readable problem descriptions.
    """
    problems: list[str] = []
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return [f"{sample_dir.name}: missing meta.json (partial/interrupted sample)"]

    meta = json.loads(meta_path.read_text())
    traj = _load(sample_dir / "latents/trajectory.pt")
    eps = _load(sample_dir / "targets/target_eps.pt")
    cond = torch.load(sample_dir / "conditioning.pt", map_location="cpu", weights_only=True)
    timesteps = json.loads((sample_dir / "timesteps.json").read_text())

    if traj.shape[0] != eps.shape[0] + 1:
        problems.append(f"{sample_dir.name}: {traj.shape[0]} latents vs {eps.shape[0]} targets")
    if len(timesteps) != eps.shape[0]:
        problems.append(f"{sample_dir.name}: {len(timesteps)} timesteps vs {eps.shape[0]} targets")
    if list(traj.shape[1:]) != meta["latent_shape"]:
        problems.append(f"{sample_dir.name}: latent shape {list(traj.shape[1:])} != meta")
    if not torch.isfinite(traj).all():
        problems.append(f"{sample_dir.name}: non-finite latents")
    if not torch.isfinite(eps).all():
        problems.append(f"{sample_dir.name}: non-finite targets")

    is_sao = "stable-audio" in str(meta.get("model_id", ""))
    cond_keys = (
        ("text_audio",)
        if is_sao
        else ("generated_prompt_embeds", "t5_prompt_embeds", "t5_attention_mask")
    )
    for key in cond_keys:
        if key not in cond:
            problems.append(f"{sample_dir.name}: conditioning missing {key}")
    if "t5_attention_mask" in cond and "t5_prompt_embeds" in cond:
        if cond["t5_attention_mask"].shape[0] != cond["t5_prompt_embeds"].shape[0]:
            problems.append(f"{sample_dir.name}: t5 mask/embed length mismatch")

    if check_step and not problems and is_sao:
        problems.extend(check_sao_step(sample_dir, meta, traj, eps, timesteps, scheduler_cache))
    elif check_step and not problems:
        key = (meta["model_id"], meta["num_inference_steps"])
        if key not in scheduler_cache:
            sched = DDIMScheduler.from_pretrained(meta["model_id"], subfolder="scheduler")
            sched.set_timesteps(meta["num_inference_steps"], device="cpu")
            scheduler_cache[key] = sched
        sched = scheduler_cache[key]

        if [int(t) for t in sched.timesteps] != timesteps:
            problems.append(f"{sample_dir.name}: cached timesteps do not match the DDIM grid")
        else:
            worst = 0.0
            for i, t in enumerate(sched.timesteps):
                got = sched.step(
                    model_output=eps[i : i + 1].float(),
                    timestep=t,
                    sample=traj[i : i + 1].float(),
                    return_dict=True,
                ).prev_sample
                worst = max(worst, (got - traj[i + 1 : i + 2].float()).abs().max().item())
            if worst > STEP_TOLERANCE:
                problems.append(
                    f"{sample_dir.name}: DDIM step invariant violated, worst={worst:.3e}"
                )
    return problems


def main(
    root_dir: str,
    check_step: int = 8,
    seed: int = 0,
) -> None:
    """Validate every sample in a trajectory dataset; spot-check the DDIM invariant.

    Args:
        root_dir: Dataset directory containing `sample_*` subdirectories.
        check_step: How many randomly chosen samples get the (slow) DDIM step-invariant check.
        seed: Seed for choosing which samples get the step check.
    """
    root = Path(root_dir).resolve()
    sample_dirs = sorted(root.glob("sample_*"))
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* directories in {root}")

    step_checked = set(random.Random(seed).sample(range(len(sample_dirs)), min(check_step, len(sample_dirs))))

    all_problems: list[str] = []
    t5_lengths: list[int] = []
    latent_shapes: set[tuple] = set()
    for idx, sample_dir in enumerate(tqdm(sample_dirs, desc="verifying")):
        problems = check_sample(sample_dir, {}, check_step=idx in step_checked)
        all_problems.extend(problems)
        if not problems:
            meta = json.loads((sample_dir / "meta.json").read_text())
            if "t5_seq_len" in meta:
                t5_lengths.append(meta["t5_seq_len"])
            latent_shapes.add(tuple(meta["latent_shape"]))

    print(f"\nsamples found      : {len(sample_dirs)}")
    print(f"healthy            : {len(sample_dirs) - len({p.split(':')[0] for p in all_problems})}")
    print(f"step-invariant run : {len(step_checked)} samples (tolerance {STEP_TOLERANCE})")
    print(f"latent shapes      : {sorted(latent_shapes)}")
    if t5_lengths:
        print(f"t5 seq len         : min={min(t5_lengths)} max={max(t5_lengths)} "
              f"(collate must pad to batch max)")
    if all_problems:
        print(f"\n{len(all_problems)} PROBLEM(S):")
        for problem in all_problems[:50]:
            print("  -", problem)
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        raise SystemExit(1)
    print("\nOK: dataset is complete and self-consistent")


if __name__ == "__main__":
    fire.Fire(main)
