# ABOUTME: Figure: the realfn adapter is accurate on its training states (forward-noised) and
# ABOUTME: wrong on deployment states (ODE inversion) — the SAME training clips, per noise level.

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/nas/lstanisz/code/lorainv/audio")
from src.inversion_lora.apply_lora import attach_inversion_lora  # noqa: E402
from src.inversion_lora.generate_real_pairs_stable_audio import (  # noqa: E402
    encode_clip,
    real_pairs,
)
from src.inversion_lora.stable_audio import ExactDPMSolver, load_teacher  # noqa: E402

CKPT = "/tmp/claude-23830/-nas-lstanisz-code-lorainv/6d9625a0-3b6e-493e-a8b4-de96103506d4/scratchpad/checkpoint_step_3000.pt"
CLIPS = sorted(Path("/nas/lstanisz/data/musiccaps/audio").glob("*.wav"))[:6]
FS = 14
device = torch.device("cuda:7")

teacher = load_teacher("stabilityai/stable-audio-open-1.0", device, 100)
solver = ExactDPMSolver(teacher.model.scheduler)
set_enabled = attach_inversion_lora(teacher.pipe.transformer, CKPT)


def student(x, index, cond):
    t = torch.tensor([solver.timesteps[index]], device=device)
    raw = teacher.forward(solver.model_input(x, index), t, cond)
    return solver.data_prediction(x, raw, index)


indist, deploy = [], []
with torch.no_grad():
    for wav in CLIPS:
        teacher.set_duration(teacher.max_duration_s)
        x0, dur = encode_clip(teacher, str(wav))
        teacher.set_duration(dur)
        cond = teacher.encode_prompt("")

        rows, targets, states, ts = real_pairs(teacher, solver, x0, cond, 500000, 16)
        inputs, tgt = rows[1:].to(device), targets.to(device)
        idx = solver.index_for(torch.tensor(ts))
        set_enabled(True)
        e_in = []
        for i in range(len(idx)):
            p = student(inputs[i:i + 1], int(idx[i]), cond)
            e_in.append(((p - tgt[i:i + 1]).norm() / tgt[i:i + 1].norm()).item())
        indist.append(e_in)

        set_enabled(False)
        xs, x = [x0], x0
        start = solver.invertible_steps
        for index in range(start - 1, -1, -1):
            x = solver.inverse(x, student(x, index + 1, cond), index)
            xs.append(x)
        xs = xs[::-1]
        e_dep = []
        for k in range(start):
            set_enabled(False)
            target = student(xs[k], k, cond)
            set_enabled(True)
            approx = student(xs[k + 1], k + 1, cond)
            e_dep.append(((approx - target).norm() / target.norm()).item())
        set_enabled(False)
        deploy.append(e_dep)

indist = torch.tensor(indist)
deploy = torch.tensor(deploy)
sig = [float(solver.sigmas[k]) for k in range(solver.invertible_steps)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
fig.suptitle("The realfn adapter is accurate where it was trained, wrong where it is deployed "
             "(same training clips)", fontsize=FS + 1, fontweight="bold")

ax = axes[0]
ax.plot(sig, indist.mean(0), "o-", ms=4, color="#55a868",
        label="on training states (forward-noised)")
ax.plot(sig, deploy.mean(0), "o-", ms=4, color="#c44e52",
        label="on deployment states (ODE inversion)")
ax.set_xscale("log")
ax.set_xlabel("noise level σ (log)", fontsize=FS)
ax.set_ylabel("adapter relative prediction error", fontsize=FS)
ax.set_title("Per noise level", fontsize=FS + 1)
ax.tick_params(labelsize=FS - 1)
ax.grid(True, linestyle="--", alpha=0.2)
ax.legend(fontsize=FS - 1)

ax = axes[1]
means = [indist.mean().item(), deploy.mean().item()]
sems = [indist.mean(1).std().item() / len(CLIPS) ** 0.5,
        deploy.mean(1).std().item() / len(CLIPS) ** 0.5]
ax.bar([0, 1], means, 0.6, yerr=sems, capsize=5, color=["#55a868", "#c44e52"],
       edgecolor="black", linewidth=0.8)
for i, m in enumerate(means):
    ax.annotate(f"{m:.3f}", (i, m), ha="center", va="bottom", fontsize=FS,
                xytext=(0, 6), textcoords="offset points")
ax.set_xticks([0, 1])
ax.set_xticklabels(["training states\n(forward-noised)", "deployment states\n(ODE inversion)"],
                   fontsize=FS - 1)
ax.set_ylabel("mean adapter relative error", fontsize=FS)
ax.set_title("Averaged over the schedule", fontsize=FS + 1)
ax.tick_params(labelsize=FS - 1)
ax.grid(True, linestyle="--", alpha=0.2, axis="y")

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path("/nas/lstanisz/code/lorainv/audio/output/realfn_diag") / \
    datetime.now().strftime("%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(out / f"realfn_indist.{ext}", dpi=150, bbox_inches="tight")
print(f"in-dist {means[0]:.3f} deploy {means[1]:.3f}")
print(f"wrote {out}")
