# ABOUTME: Latent reconstruction (invert with adapter, denoise with frozen teacher) on MusicCaps
# ABOUTME: (realfn's training audio) vs MedleyDB (benchmark), for no-LoRA / trajectory / realfn.

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

SCRATCH = "/tmp/claude-23830/-nas-lstanisz-code-lorainv/6d9625a0-3b6e-493e-a8b4-de96103506d4/scratchpad"
REALFN = f"{SCRATCH}/checkpoint_step_3000.pt"
TRAJ = f"{SCRATCH}/checkpoint_step_4000.pt"
N = 10
MUSICCAPS = sorted(Path("/nas/lstanisz/data/musiccaps/audio").glob("*.wav"))[:N]
MEDLEY = sorted(Path("/nas/lstanisz/data/medleydb/V1_mix").glob("*/*_MIX.wav"))[:N]
SOURCES = {"MusicCaps\n(realfn training audio)": MUSICCAPS, "MedleyDB\n(benchmark)": MEDLEY}
FS = 14
device = torch.device("cuda:7")

teacher = load_teacher("stabilityai/stable-audio-open-1.0", device, 100)
solver = ExactDPMSolver(teacher.model.scheduler)
cond = None


def predict(x, index):
    t = torch.tensor([solver.timesteps[index]], device=device)
    raw = teacher.forward(solver.model_input(x, index), t, cond)
    return solver.data_prediction(x, raw, index)


@torch.no_grad()
def recon_error(x0, set_enabled, use_adapter):
    steps = solver.invertible_steps
    if set_enabled:
        set_enabled(use_adapter)
    x_t = ode_invert(solver, x0, predict, steps)
    if set_enabled:
        set_enabled(False)
    x_rec = ode_denoise(solver, x_t, solver.invertible_steps - steps, predict)
    return ((x_rec - x0).norm() / x0.norm()).item()


@torch.no_grad()
def encode_all():
    out = {}
    for name, wavs in SOURCES.items():
        xs = []
        for wav in wavs:
            teacher.set_duration(teacher.max_duration_s)
            x0, dur = encode_clip(teacher, str(wav))
            xs.append((x0, dur))
        out[name] = xs
    return out


encoded = encode_all()
cond = teacher.encode_prompt("")  # empty caption for a pure reconstruction round trip

results = {}  # (source) -> {method: [errors]}
for name in SOURCES:
    results[name] = {"no LoRA": [], "trajectory LoRA": [], "realfn LoRA": []}

# no-LoRA and realfn share one attach (realfn); trajectory needs its own.
set_realfn = attach_inversion_lora(teacher.pipe.transformer, REALFN)
for name, xs in encoded.items():
    for x0, dur in xs:
        teacher.set_duration(dur)
        results[name]["no LoRA"].append(recon_error(x0, set_realfn, False))
        results[name]["realfn LoRA"].append(recon_error(x0, set_realfn, True))
set_realfn(False)

# fresh teacher for the trajectory adapter (same injected name "inversion")
teacher2 = load_teacher("stabilityai/stable-audio-open-1.0", device, 100)
solver2 = ExactDPMSolver(teacher2.model.scheduler)
set_traj = attach_inversion_lora(teacher2.pipe.transformer, TRAJ)
cond2 = teacher2.encode_prompt("")


def predict2(x, index):
    t = torch.tensor([solver2.timesteps[index]], device=device)
    raw = teacher2.forward(solver2.model_input(x, index), t, cond2)
    return solver2.data_prediction(x, raw, index)


@torch.no_grad()
def recon_traj(x0):
    steps = solver2.invertible_steps
    set_traj(True)
    x_t = ode_invert(solver2, x0, predict2, steps)
    set_traj(False)
    x_rec = ode_denoise(solver2, x_t, solver2.invertible_steps - steps, predict2)
    return ((x_rec - x0).norm() / x0.norm()).item()


for name, xs in encoded.items():
    for x0, dur in xs:
        teacher2.set_duration(dur)
        results[name]["trajectory LoRA"].append(recon_traj(x0))

# DDPM-inversion is exact by construction (stores per-step noise); latent error ~ 0 on any audio.
METHODS = ["DDPM-inv", "no LoRA", "trajectory LoRA", "realfn LoRA"]
COLORS = {"DDPM-inv": "#2ca02c", "no LoRA": "#7f7f7f",
          "trajectory LoRA": "#4c72b0", "realfn LoRA": "#c44e52"}

fig, ax = plt.subplots(figsize=(11, 6))
width = 0.38
xpos = range(len(METHODS))
for s, (name, _) in enumerate(SOURCES.items()):
    vals, errs = [], []
    for m in METHODS:
        if m == "DDPM-inv":
            vals.append(0.0)
            errs.append(0.0)
        else:
            v = torch.tensor(results[name][m])
            vals.append(v.mean().item())
            errs.append(v.std().item() / len(v) ** 0.5)
    off = (s - 0.5) * width
    bars = ax.bar([x + off for x in xpos], vals, width, yerr=errs, capsize=4,
                  edgecolor="black", linewidth=0.8, label=name.replace("\n", " "),
                  color=[COLORS[m] for m in METHODS], alpha=0.75 if s else 1.0,
                  hatch="//" if s else None)
    for x, v in zip(xpos, vals):
        tag = "≈0 (exact)" if v == 0 else f"{v:.2f}"
        ax.annotate(tag, (x + off, v), ha="center", va="bottom", fontsize=FS - 3,
                    xytext=(0, 3), textcoords="offset points")
ax.set_xticks(list(xpos))
ax.set_xticklabels(METHODS, fontsize=FS)
ax.set_ylabel("relative latent reconstruction error (lower = better)", fontsize=FS)
ax.set_title("Real-audio reconstruction: training-audio source vs benchmark source\n"
             "(solid = MusicCaps, hatched = MedleyDB)", fontsize=FS + 1, fontweight="bold")
ax.tick_params(labelsize=FS - 1)
ax.grid(True, linestyle="--", alpha=0.2, axis="y")
ax.legend(fontsize=FS - 1)
fig.tight_layout()

out = Path("/nas/lstanisz/code/lorainv/audio/output/recon_by_source") / \
    datetime.now().strftime("%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(out / f"recon_by_source.{ext}", dpi=150, bbox_inches="tight")
for name in SOURCES:
    print(name.replace("\n", " "), {m: round(torch.tensor(results[name][m]).mean().item(), 3)
                                     for m in ["no LoRA", "trajectory LoRA", "realfn LoRA"]})
print(f"wrote {out}")
