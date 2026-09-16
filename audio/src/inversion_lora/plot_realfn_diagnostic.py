# ABOUTME: Figure for the realfn reconstruction collapse: per-step shift-gap error over the whole
# ABOUTME: grid (no-adapter tiny vs realfn over-corrects) and the deployment round-trip errors.

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/nas/lstanisz/code/lorainv/audio")
from src.inversion_lora.apply_lora import attach_inversion_lora  # noqa: E402
from src.inversion_lora.generate_real_pairs_stable_audio import encode_clip  # noqa: E402
from src.inversion_lora.stable_audio import (  # noqa: E402
    ExactDPMSolver,
    load_teacher,
    ode_denoise,
    ode_invert,
)

CKPT = "/tmp/claude-23830/-nas-lstanisz-code-lorainv/6d9625a0-3b6e-493e-a8b4-de96103506d4/scratchpad/checkpoint_step_3000.pt"
CLIPS = sorted(Path("/nas/lstanisz/data/musiccaps/audio").glob("*.wav"))[:6]
FS = 14
device = torch.device("cuda:7")

teacher = load_teacher("stabilityai/stable-audio-open-1.0", device, 100)
solver = ExactDPMSolver(teacher.model.scheduler)
set_enabled = attach_inversion_lora(teacher.pipe.transformer, CKPT)
text_audio = teacher.encode_prompt("")


def predict(x, index):
    t = torch.tensor([solver.timesteps[index]], device=device)
    raw = teacher.forward(solver.model_input(x, index), t, text_audio)
    return solver.data_prediction(x, raw, index)


@torch.no_grad()
def roundtrip(x0, invert_with_adapter):
    steps = solver.invertible_steps
    set_enabled(invert_with_adapter)
    x_t = ode_invert(solver, x0, predict, steps)
    set_enabled(False)
    x_rec = ode_denoise(solver, x_t, solver.invertible_steps - steps, predict)
    return ((x_rec - x0).norm() / x0.norm()).item()


rt_no, rt_re, perstep_no, perstep_re = [], [], [], []
with torch.no_grad():
    for wav in CLIPS:
        teacher.set_duration(teacher.max_duration_s)
        x0, dur = encode_clip(teacher, str(wav))
        teacher.set_duration(dur)
        rt_no.append(roundtrip(x0, False))
        rt_re.append(roundtrip(x0, True))

        # per-step gap along the real no-LoRA inversion trajectory
        set_enabled(False)
        xs, x = [x0], x0
        start = solver.invertible_steps
        for index in range(start - 1, -1, -1):
            x = solver.inverse(x, predict(x, index + 1), index)
            xs.append(x)
        xs = xs[::-1]
        e_no, e_re = [], []
        for k in range(start):
            target = predict(xs[k], k)
            set_enabled(False)
            a_no = predict(xs[k + 1], k + 1)
            set_enabled(True)
            a_re = predict(xs[k + 1], k + 1)
            set_enabled(False)
            e_no.append(((a_no - target).norm() / target.norm()).item())
            e_re.append(((a_re - target).norm() / target.norm()).item())
        perstep_no.append(e_no)
        perstep_re.append(e_re)

perstep_no = torch.tensor(perstep_no).mean(0)
perstep_re = torch.tensor(perstep_re).mean(0)
sigmas = [float(solver.sigmas[k]) for k in range(solver.invertible_steps)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
fig.suptitle("Why the real-forward-noise adapter breaks inversion: it corrects a gap that isn't "
             "there", fontsize=FS + 2, fontweight="bold")

ax = axes[0]
ax.plot(sigmas, perstep_no, "o-", ms=4, color="#7f7f7f",
        label="no adapter (true shift gap)")
ax.plot(sigmas, perstep_re, "o-", ms=4, color="#c44e52",
        label="realfn adapter (its correction)")
ax.set_xscale("log")
ax.set_xlabel("noise level σ (log)", fontsize=FS)
ax.set_ylabel("relative prediction error vs teacher target", fontsize=FS)
ax.set_title("Per-step, on the real inversion trajectory", fontsize=FS + 1)
ax.tick_params(labelsize=FS - 1)
ax.grid(True, linestyle="--", alpha=0.2)
ax.legend(fontsize=FS - 1)

ax = axes[1]
means = [sum(rt_no) / len(rt_no), sum(rt_re) / len(rt_re)]
sems = [torch.tensor(rt_no).std().item() / len(rt_no) ** 0.5,
        torch.tensor(rt_re).std().item() / len(rt_re) ** 0.5]
ax.bar([0, 1], means, 0.6, yerr=sems, capsize=5,
       color=["#7f7f7f", "#c44e52"], edgecolor="black", linewidth=0.8)
for i, m in enumerate(means):
    ax.annotate(f"{m:.3f}", (i, m), ha="center", va="bottom", fontsize=FS,
                xytext=(0, 6), textcoords="offset points")
ax.set_xticks([0, 1])
ax.set_xticklabels(["no LoRA", "realfn LoRA"], fontsize=FS)
ax.set_ylabel("relative latent error", fontsize=FS)
ax.set_title("Deployment round trip (invert w/ adapter,\ndenoise w/ frozen teacher)",
             fontsize=FS + 1)
ax.tick_params(labelsize=FS - 1)
ax.grid(True, linestyle="--", alpha=0.2, axis="y")

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path("/nas/lstanisz/code/lorainv/audio/output/realfn_diag") / \
    datetime.now().strftime("%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(out / f"realfn_diagnostic.{ext}", dpi=150, bbox_inches="tight")
print(f"round-trip no-LoRA {means[0]:.3f} realfn {means[1]:.3f}")
print(f"peak per-step: no-adapter {perstep_no.max():.3f} realfn {perstep_re.max():.3f}")
print(f"wrote {out}")
