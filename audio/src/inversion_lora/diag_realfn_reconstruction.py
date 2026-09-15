# ABOUTME: Decisive diagnostic for the realfn reconstruction collapse: latent round-trip with the
# ABOUTME: adapter on vs off, and per-step prediction error on the real deployment trajectory.

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/nas/lstanisz/code/lorainv/audio")
from src.inversion_lora.apply_lora import attach_inversion_lora  # noqa: E402
from src.inversion_lora.generate_real_pairs_stable_audio import encode_clip  # noqa: E402
from src.inversion_lora.stable_audio import (  # noqa: E402
    ExactDPMSolver,
    load_teacher,
    ode_denoise,
    ode_invert,
)

CKPT = "PATH_TO/saocos_realfn_r8_a4_lr5e-5/checkpoint_step_3000.pt"  # set before running
CLIPS = sorted(Path("/nas/lstanisz/data/musiccaps/audio").glob("*.wav"))[:4]
device = torch.device("cuda:7")

teacher = load_teacher("stabilityai/stable-audio-open-1.0", device, 100)
solver = ExactDPMSolver(teacher.model.scheduler)
set_enabled = attach_inversion_lora(teacher.pipe.transformer, CKPT)

# Unconditional conditioning: reconstruction denoises with the source caption, but here we hold
# the caption fixed (empty) so the round trip isolates the inversion, not caption effects.
text_audio = teacher.encode_prompt("")


def predict(x, index):
    """Data prediction at (x, timesteps[index]); adapter state set by the caller."""
    t = torch.tensor([solver.timesteps[index]], device=device)
    raw = teacher.forward(solver.model_input(x, index), t, text_audio)
    return solver.data_prediction(x, raw, index)


@torch.no_grad()
def roundtrip(x0, invert_with_adapter):
    """Deployment round trip: invert (adapter per flag), denoise with the FROZEN teacher."""
    steps = solver.invertible_steps
    set_enabled(invert_with_adapter)
    x_t = ode_invert(solver, x0, predict, steps)
    set_enabled(False)  # denoise is always the frozen teacher, as in the real eval
    x_rec = ode_denoise(solver, x_t, solver.invertible_steps - steps, predict)
    return ((x_rec - x0).norm() / x0.norm()).item()


print(f"{'clip':<14} {'no-LoRA rel':>12} {'realfn rel':>12}  (invert-only adapter, as deployed)")
rels = {"nolora": [], "realfn": []}
for wav in CLIPS:
    teacher.set_duration(teacher.max_duration_s)
    x0, dur = encode_clip(teacher, str(wav))
    teacher.set_duration(dur)
    with torch.no_grad():
        r_no = roundtrip(x0, invert_with_adapter=False)
        r_re = roundtrip(x0, invert_with_adapter=True)
    rels["nolora"].append(r_no)
    rels["realfn"].append(r_re)
    print(f"{wav.name[:14]:<14} {r_no:>12.4f} {r_re:>12.4f}")
print(f"{'MEAN':<14} {sum(rels['nolora'])/len(CLIPS):>12.4f} "
      f"{sum(rels['realfn'])/len(CLIPS):>12.4f}")

# Per-step: along the true no-LoRA inversion trajectory, how well does the realfn student predict
# the teacher's target at the noisier point, vs the plain (no-adapter) shift approximation?
print("\nper-step shift-gap prediction error on the real deployment trajectory (clip 0):")
teacher.set_duration(teacher.max_duration_s)
x0, dur = encode_clip(teacher, str(CLIPS[0]))
teacher.set_duration(dur)
with torch.no_grad():
    set_enabled(False)
    # Rebuild the deployment trajectory (no-LoRA inversion), keeping every state.
    xs = [x0]
    x = x0
    start = solver.invertible_steps
    for index in range(start - 1, -1, -1):
        x = solver.inverse(x, predict(x, index + 1), index)
        xs.append(x)
    xs = xs[::-1]  # xs[k] ~ latent at grid point k (noisy..clean)
    # target at the noisier point k: D(xs[k], t_k). approximations query the cleaner point k+1.
    for k in [90, 60, 30, 5]:
        target = predict(xs[k], k)  # teacher D at the noisier point (the thing inversion needs)
        set_enabled(False)
        approx_noadapt = predict(xs[k + 1], k + 1)
        set_enabled(True)
        approx_realfn = predict(xs[k + 1], k + 1)
        set_enabled(False)
        e_no = ((approx_noadapt - target).norm() / target.norm()).item()
        e_re = ((approx_realfn - target).norm() / target.norm()).item()
        print(f"  grid k={k:>2} (sigma {float(solver.sigmas[k]):.3g}): "
              f"no-adapter shift err {e_no:.4f} | realfn err {e_re:.4f}")
