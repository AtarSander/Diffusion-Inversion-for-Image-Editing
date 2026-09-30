# ABOUTME: Compare sampling-ODE states with forward-diffused copies of the same final sample, level
# ABOUTME: by level, on AudioLDM2 and Stable Audio Open: marginal match, coupling, one-step gap.

import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from loguru import logger

AUDIO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_EDITING_CODE = AUDIO_ROOT / "editing/AudioEditingCode/code"
for p in (AUDIO_ROOT, AUDIO_EDITING_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CAPTIONS_CSV = AUDIO_ROOT / "editing/music_caps_dl/metadata/musiccaps-public.csv"
MODEL_IDS = {"audioldm2": "cvssp/audioldm2-large", "stable_audio": "stabilityai/stable-audio-open-1.0"}
AUDIOLDM2_LENGTH_S = 10.24  # the adapter's training window, latent [8, 256, 16]
# Reserved for this analysis. Training uses 42..1541, eval inputs 314159.., real pairs 500000..
SEED_BASE = 910_000
NOISE_SEED_OFFSET = 1_000_000  # forward-diffusion draws, one generator per sample
PROJ_SEED = 1234  # the random projection shared by every shard, so features pool across them
# Per-level scalars, each for the ODE state and for the forward-diffused state of the same x0.
PAIRED = ("rms", "nstd", "nkurt", "nlagt", "nlagf", "cosx0")
ODE_ONLY = ("cosz", "straight")


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    """Cosine similarity of two tensors, flattened, in float64."""
    u, v = u.double().flatten(), v.double().flatten()
    return float(u @ v / (u.norm() * v.norm()).clamp_min(1e-300))


def noise_stats(n: torch.Tensor, time_dim: int, freq_dim: int | None) -> dict[str, float]:
    """Gaussianity and local structure of one effective-noise tensor (no batch dimension).

    iid N(0, 1) noise gives std 1, excess kurtosis 0 and lag-1 autocorrelations 0; the forward
    draws are exactly that, so their values are the null the ODE values are read against.

    Args:
        n: Effective noise of one state.
        time_dim: Axis of `n` that runs along time.
        freq_dim: Axis that runs along frequency, or None when the latent has no such axis.

    Returns:
        `std`, excess kurtosis `kurt`, and lag-1 autocorrelation along time and frequency.
    """
    c = n.double() - n.double().mean()
    var = c.pow(2).mean()

    def lag1(dim: int) -> float:
        a = c.narrow(dim, 0, c.shape[dim] - 1)
        b = c.narrow(dim, 1, c.shape[dim] - 1)
        return float((a * b).mean() / var)

    return {
        "std": float(var.sqrt()),
        "kurt": float(c.pow(4).mean() / var.pow(2) - 3.0),
        "lag_t": lag1(time_dim),
        "lag_f": lag1(freq_dim) if freq_dim is not None else float("nan"),
    }


def level_metrics(
    x: torch.Tensor, y: torch.Tensor, x0: torch.Tensor, z_hat: torch.Tensor, a: float, s: float,
    time_dim: int, freq_dim: int | None,
) -> dict[str, float]:
    """Every per-level scalar for one sample: the ODE state `x` against the forward state `y`.

    Both states sit at the level with forward coefficients `(a, s)`, i.e. `y = a x0 + s eps`. The
    effective noise of a state is `(state - a x0) / s`, which for `y` is `eps` itself. The straight
    reference `a x0 + s z_hat` uses the trajectory's own starting noise, so its distance to `x`
    measures how far the ODE path bends away from the line between its endpoints.

    Args:
        x: ODE state at this level.
        y: Forward-diffused state of the same final sample at this level.
        x0: The trajectory's final (clean) latent.
        z_hat: The trajectory's starting noise divided by the first level's `s`.
        a: Signal coefficient of this level.
        s: Noise coefficient of this level.
        time_dim: Time axis of one state.
        freq_dim: Frequency axis of one state, or None.

    Returns:
        Flat dict of `<name>_ode`, `<name>_fwd` (names in PAIRED) and the ODE-only names.
    """
    out: dict[str, float] = {}
    for tag, state in (("ode", x), ("fwd", y)):
        n = (state - a * x0) / s
        st = noise_stats(n, time_dim, freq_dim)
        out[f"rms_{tag}"] = float(state.double().pow(2).mean().sqrt())
        out[f"nstd_{tag}"] = st["std"]
        out[f"nkurt_{tag}"] = st["kurt"]
        out[f"nlagt_{tag}"] = st["lag_t"]
        out[f"nlagf_{tag}"] = st["lag_f"]
        out[f"cosx0_{tag}"] = cosine(n, x0)
        if tag == "ode":
            out["cosz"] = cosine(n, z_hat)
    straight = a * x0 + s * z_hat
    out["straight"] = float((x - straight).double().norm() / straight.double().norm())
    return out


def mmd_test(
    fx: torch.Tensor, fy: torch.Tensor, num_permutations: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """Two-sample MMD test with the median-distance bandwidth and a permutation p-value.

    Args:
        fx: Features `[m, k]` of the first sample.
        fy: Features `[n, k]` of the second sample.
        num_permutations: Label permutations for the null.
        seed: Seed of the permutations.

    Returns:
        `(mmd2, p_value, bandwidth)`.
    """
    fx, fy = fx.double(), fy.double()
    pooled = torch.cat([fx, fy])
    d = torch.cdist(pooled, pooled)
    bandwidth = float(d[torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)].median())
    k = torch.exp(-d.pow(2) / (2 * bandwidth**2))
    m = fx.shape[0]

    def stat(idx: torch.Tensor) -> float:
        kk = k[idx][:, idx]
        n = kk.shape[0] - m
        kxx, kyy, kxy = kk[:m, :m], kk[m:, m:], kk[:m, m:]
        return float(
            (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
            + (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
            - 2 * kxy.mean()
        )

    observed = stat(torch.arange(pooled.shape[0]))
    gen = torch.Generator().manual_seed(seed)
    null = [stat(torch.randperm(pooled.shape[0], generator=gen)) for _ in range(num_permutations)]
    p = (1 + sum(v >= observed for v in null)) / (1 + num_permutations)
    return observed, p, bandwidth


class AudioLDM2Probe:
    """AudioLDM2 DDIM at w=1: the trajectory generator's own sampler and the adapter's pairing.

    The adapter's pair is (cleaner latent, the NOISIER timestep) -> the teacher's epsilon at the
    noisier latent, so the LoRA-off gap of pair i is `||eps(x_{i+1}, t_i) - eps(x_i, t_i)||^2`,
    the quantity the trainer logs as its LoRA-disabled loss. The forward version is built exactly
    as `generate_real_pairs_audioldm2.real_pairs` builds its (realfn) pairs.
    """

    time_dim, freq_dim = 1, 2  # one state is [C, T, F]

    def __init__(self, device: torch.device, steps: int):
        from models import load_model

        self.device = device
        self.ldm = load_model(MODEL_IDS["audioldm2"], device, steps, edit_method="ddim")
        self.ldm.model.unet.eval()
        self.scheduler = self.ldm.model.scheduler
        self.grid = list(self.scheduler.timesteps)
        self.abar = self.scheduler.alphas_cumprod.detach().double().cpu()

    def sample(self, prompt: str, seed: int) -> dict[str, Any]:
        """One w=1 trajectory `[N+1, C, T, F]`, noisiest first, ending at the clean latent."""
        from src.inversion_lora.generate_trajectories import sample_with_trajectory

        out = sample_with_trajectory(
            self.ldm, prompt, guidance_scale=1.0, audio_length_in_s=AUDIOLDM2_LENGTH_S,
            seed=seed, save_uncond_target=False,
        )
        hidden, t5_embeds, t5_mask = self.ldm.encode_text([prompt])
        self.cond = {"encoder_hidden_states": hidden, "class_labels": t5_embeds,
                     "encoder_attention_mask": t5_mask}
        self.eps_cached = out["target_eps"].float()
        assert out["timesteps"] == [int(t) for t in self.grid], "sampler and probe grids differ"
        abar = self.abar[out["timesteps"]]
        return {"trajectory": out["trajectory"].float(), "a": abar.sqrt().tolist(),
                "s": (1 - abar).sqrt().tolist(), "levels": out["timesteps"]}

    @property
    def num_pairs(self) -> int:
        return len(self.grid)

    def _eps(self, x: torch.Tensor, i: int) -> torch.Tensor:
        t = self.grid[i]
        xb = x.to(self.device, dtype=self.ldm.model.unet.dtype)
        return self.ldm.unet_forward(
            self.scheduler.scale_model_input(xb, t), timestep=t, **self.cond
        )[0].sample

    @torch.no_grad()
    def ode_gap(self, trajectory: torch.Tensor, i: int) -> float:
        cleaner = self._eps(trajectory[i + 1].unsqueeze(0), i).float().cpu()[0]
        return float((cleaner - self.eps_cached[i]).pow(2).mean())

    @torch.no_grad()
    def forward_gap(self, y: torch.Tensor, i: int) -> float:
        yb = y.unsqueeze(0).to(self.device, dtype=self.ldm.model.unet.dtype)
        eps_hat = self._eps(yb, i)
        x_clean = self.scheduler.step(eps_hat, self.grid[i], yb, eta=0).prev_sample
        return float((self._eps(x_clean, i) - eps_hat).float().pow(2).mean())


class StableAudioProbe:
    """Stable Audio Open on the native cosine grid with the first-order ODE the adapter uses.

    The adapter's pair is matched-timestep in data-prediction space, so the LoRA-off gap of pair i
    is `||D(x_{i+1}, t_{i+1}) - D(x_i, t_i)||^2`: both predictions are made while sampling, so the
    ODE gap is free. The forward version mirrors `generate_real_pairs_stable_audio`. The last step
    ends at sigma = 0 and has no pair, hence one pair fewer than levels.
    """

    time_dim, freq_dim = 1, None  # one state is [C, L]

    def __init__(self, device: torch.device, steps: int):
        from src.inversion_lora.stable_audio import ExactDPMSolver, load_teacher

        self.device = device
        self.teacher = load_teacher(MODEL_IDS["stable_audio"], device, steps, schedule="cosine")
        self.solver = ExactDPMSolver(self.teacher.model.scheduler)

    def sample(self, prompt: str, seed: int) -> dict[str, Any]:
        """One w=1 trajectory `[N+1, C, L]`, noisiest first, ending at sigma = 0."""
        self.text_audio = self.teacher.encode_prompt(prompt)
        trajectory, data, grid, _ = self.teacher.ode_trajectory(
            self.text_audio, seed=seed, progress=False
        )
        self.data = data.float()
        coeffs = [self.solver._alpha_sigma(i) for i in range(len(grid))]
        return {"trajectory": trajectory.float(), "a": [float(c[0]) for c in coeffs],
                "s": [float(c[1]) for c in coeffs], "levels": [float(t) for t in grid]}

    @property
    def num_pairs(self) -> int:
        return len(self.solver.timesteps) - 1

    def _data(self, x: torch.Tensor, i: int) -> torch.Tensor:
        t = torch.tensor([self.solver.timesteps[i]], device=self.device)
        raw = self.teacher.forward(self.solver.model_input(x, i), t, self.text_audio)
        return self.solver.data_prediction(x, raw, i)

    def ode_gap(self, trajectory: torch.Tensor, i: int) -> float:
        return float((self.data[i + 1] - self.data[i]).pow(2).mean())

    @torch.no_grad()
    def forward_gap(self, y: torch.Tensor, i: int) -> float:
        yb = y.unsqueeze(0).to(self.device)
        d_noisier = self._data(yb, i)
        cleaner = self.solver.forward(yb, d_noisier, i)
        return float((self._data(cleaner, i + 1) - d_noisier).float().pow(2).mean())


def source_revision() -> str:
    """The commit this code came from, or "unknown" when run from an archive without .git."""
    from src.inversion_lora.generate_trajectories import git_sha

    try:
        return git_sha()
    except Exception:  # noqa: BLE001 -- provenance only; never worth failing a run over
        return "unknown"


PROBES = {"audioldm2": AudioLDM2Probe, "stable_audio": StableAudioProbe}


def mmd_level_indices(num_levels: int, every: int) -> list[int]:
    """Levels whose projected states are kept for the two-sample test: a stride plus the last."""
    idx = list(range(0, num_levels, every))
    return idx if idx[-1] == num_levels - 1 else idx + [num_levels - 1]


def run(
    model: str,
    shard: int = 0,
    num_shards: int = 1,
    num_samples: int = 1000,
    start_index: int = 2000,
    steps: int = 200,
    device: str = "cuda:0",
    out_root: str = "output/ode_vs_forward",
    gap_stride: int = 4,
    gap_every: int = 2,
    mmd_every: int = 5,
    proj_dim: int = 128,
    keep_trajectories: int = 2,
) -> None:
    """Sample one shard of w=1 trajectories and score every level against forward diffusion.

    Samples are caption rows `start_index .. start_index + num_samples` (held out from every
    adapter's training captions), shard `k` taking every `num_shards`-th one. The teacher's gaps
    are only computed on samples whose index is a multiple of `gap_stride`, the forward one only
    at every `gap_every`-th level.

    Args:
        model: `audioldm2` or `stable_audio`.
        shard: This shard's index.
        num_shards: Number of shards the samples are split over.
        num_samples: Total samples across all shards.
        start_index: First MusicCaps caption row.
        steps: Sampling steps.
        device: Torch device.
        out_root: Output root; shards land in `<out_root>/<model>/`.
        gap_stride: Every `gap_stride`-th sample gets the model-based gaps.
        gap_every: Level stride of the forward-state gap.
        mmd_every: Level stride of the kept projected states.
        proj_dim: Dimension of the random projection the two-sample test runs on.
        keep_trajectories: Full trajectories of this shard to save for later inspection.
    """
    from dotenv import load_dotenv

    # HF_HOME / HF_TOKEN (Stable Audio Open is gated) come from the ignored audio/.env, as in the
    # trajectory generators; loaded before any model import so the hub cache path takes effect.
    load_dotenv(AUDIO_ROOT / ".env", override=True)
    from src.inversion_lora.generate_trajectories import load_captions

    assert model in PROBES, f"model must be one of {sorted(PROBES)}, got {model!r}"
    assert 0 <= shard < num_shards, (shard, num_shards)
    out_dir = AUDIO_ROOT / out_root / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"shard_{shard:03d}_of_{num_shards:03d}.pt"
    if out_file.exists():
        logger.info("{} exists; nothing to do", out_file)
        return

    records = load_captions(CAPTIONS_CSV, "caption")
    indices = list(range(start_index, start_index + num_samples))[shard::num_shards]
    assert indices and indices[-1] < len(records), (start_index, num_samples, len(records))

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    probe = PROBES[model](dev, steps)

    rows: dict[str, list[list[float]]] = {}
    gap_ode: list[list[float]] = []
    gap_fwd: list[list[float]] = []
    feats_ode: list[torch.Tensor] = []
    feats_fwd: list[torch.Tensor] = []
    proj = None
    meta: dict[str, Any] = {}
    kept = 0
    for count, sample_idx in enumerate(indices):
        tic = time.time()
        seed = SEED_BASE + sample_idx
        traj = probe.sample(records[sample_idx]["prompt"], seed)
        trajectory, a, s = traj["trajectory"], traj["a"], traj["s"]
        num_levels = trajectory.shape[0] - 1
        assert len(a) == len(s) == num_levels, (len(a), len(s), num_levels)
        x0, z_hat = trajectory[-1], trajectory[0] / s[0]
        if proj is None:
            dim = x0.numel()
            proj = torch.randn(dim, proj_dim, generator=torch.Generator().manual_seed(PROJ_SEED))
            proj /= math.sqrt(dim)
            mmd_levels = mmd_level_indices(num_levels, mmd_every)
            meta = {"model": model, "model_id": MODEL_IDS[model], "steps": steps,
                    "levels": traj["levels"], "a": a, "s": s, "mmd_levels": mmd_levels,
                    "state_shape": list(x0.shape), "seed_base": SEED_BASE,
                    "noise_seed_offset": NOISE_SEED_OFFSET, "proj_seed": PROJ_SEED,
                    "proj_dim": proj_dim, "gap_stride": gap_stride, "gap_every": gap_every,
                    "start_index": start_index, "num_samples": num_samples, "shard": shard,
                    "num_shards": num_shards, "git_sha": source_revision()}
        gen = torch.Generator().manual_seed(NOISE_SEED_OFFSET + seed)
        with_gap = sample_idx % gap_stride == 0
        sample_rows: dict[str, list[float]] = {}
        g_ode = [float("nan")] * num_levels
        g_fwd = [float("nan")] * num_levels
        f_ode, f_fwd = [], []
        for i in range(num_levels):
            y = a[i] * x0 + s[i] * torch.randn(x0.shape, generator=gen)
            for k, v in level_metrics(trajectory[i], y, x0, z_hat, a[i], s[i],
                                      probe.time_dim, probe.freq_dim).items():
                sample_rows.setdefault(k, []).append(v)
            if i in mmd_levels:
                f_ode.append(trajectory[i].flatten() @ proj)
                f_fwd.append(y.flatten() @ proj)
            if i < probe.num_pairs and (with_gap or model == "stable_audio"):
                g_ode[i] = probe.ode_gap(trajectory, i)  # free on Stable Audio
            if with_gap and i < probe.num_pairs and i % gap_every == 0:
                g_fwd[i] = probe.forward_gap(y, i)
        for k, v in sample_rows.items():
            rows.setdefault(k, []).append(v)
        gap_ode.append(g_ode)
        gap_fwd.append(g_fwd)
        feats_ode.append(torch.stack(f_ode))
        feats_fwd.append(torch.stack(f_fwd))
        if kept < keep_trajectories:
            torch.save({"trajectory": trajectory, "sample_idx": sample_idx, "seed": seed},
                       out_dir / f"trajectory_{sample_idx:06d}.pt")
            kept += 1
        logger.info("shard {} sample {}/{} (row {}, gaps={}) {:.1f}s", shard, count + 1,
                    len(indices), sample_idx, with_gap, time.time() - tic)

    torch.save({
        "meta": meta,
        "sample_idx": indices,
        "metrics": {k: torch.tensor(v, dtype=torch.float64) for k, v in rows.items()},
        "gap_ode": torch.tensor(gap_ode, dtype=torch.float64),
        "gap_fwd": torch.tensor(gap_fwd, dtype=torch.float64),
        "feats_ode": torch.stack(feats_ode),
        "feats_fwd": torch.stack(feats_fwd),
    }, out_file)
    logger.info("wrote {}", out_file)


def markdown_table(df, fmt: str = ".4g") -> str:
    """A pipe table of a DataFrame, formatting floats with `fmt` (no tabulate dependency)."""
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else format(float(v), fmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---:" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-level nanmean, SEM and count over samples (axis 0)."""
    n = np.sum(~np.isnan(x), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=0)
        sem = np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem, n


def aggregate(
    model: str, out_root: str = "output/ode_vs_forward", num_permutations: int = 200,
    report_levels: int = 12,
) -> None:
    """Merge a model's shards into `levels.csv` and a `REPORT.md` table.

    Args:
        model: `audioldm2` or `stable_audio`.
        out_root: Root the shards were written under.
        num_permutations: Permutations per level for the MMD p-value.
        report_levels: Evenly spaced levels shown in the report table.
    """
    import pandas as pd

    out_dir = AUDIO_ROOT / out_root / model
    shards = sorted(out_dir.glob("shard_*_of_*.pt"))
    assert shards, f"no shards under {out_dir}"
    parts = [torch.load(p, weights_only=True) for p in shards]
    totals = {int(p.name.split("_of_")[1].split(".")[0]) for p in shards}
    assert len(totals) == 1 and len(shards) == totals.pop(), (
        f"expected every shard of one split, found {[p.name for p in shards]}"
    )
    meta = parts[0]["meta"]
    for p in parts[1:]:
        for key in ("levels", "a", "s", "mmd_levels", "steps", "model_id"):
            assert p["meta"][key] == meta[key], f"shards disagree on {key}"
    metrics = {k: torch.cat([p["metrics"][k] for p in parts]).numpy() for k in parts[0]["metrics"]}
    gap_ode = torch.cat([p["gap_ode"] for p in parts]).numpy()
    gap_fwd = torch.cat([p["gap_fwd"] for p in parts]).numpy()
    feats_ode = torch.cat([p["feats_ode"] for p in parts])
    feats_fwd = torch.cat([p["feats_fwd"] for p in parts])
    n_samples = gap_ode.shape[0]
    num_levels = len(meta["levels"])

    table: dict[str, Any] = {"level": np.arange(num_levels), "timestep": meta["levels"],
                             "a": meta["a"], "s": meta["s"],
                             "sigma_equiv": np.array(meta["s"]) / np.array(meta["a"])}
    for name, values in metrics.items():
        table[f"{name}_mean"], table[f"{name}_sem"], _ = _mean_sem(values)
    # Gaps compared on the same samples and levels: both are defined only where gap_fwd is.
    both = ~np.isnan(gap_fwd) & ~np.isnan(gap_ode)
    g_o, g_f = np.where(both, gap_ode, np.nan), np.where(both, gap_fwd, np.nan)
    table["gap_ode_mean"], table["gap_ode_sem"], table["gap_n"] = _mean_sem(g_o)
    table["gap_fwd_mean"], table["gap_fwd_sem"], _ = _mean_sem(g_f)
    table["gap_ratio"] = table["gap_fwd_mean"] / table["gap_ode_mean"]
    table["mmd2"] = np.full(num_levels, np.nan)
    table["mmd_p"] = np.full(num_levels, np.nan)
    # ODE and forward states of the SAME sample are near-twins at low noise, which biases a paired
    # test toward "no difference" (p -> 1). Compare ODE states of one half of the samples with
    # forward states of the other half, so the two sets are independent draws.
    half_a, half_b = torch.arange(0, n_samples, 2), torch.arange(1, n_samples, 2)
    for j, level in enumerate(meta["mmd_levels"]):
        mmd2, p, _ = mmd_test(feats_ode[half_a, j], feats_fwd[half_b, j], num_permutations,
                              seed=level)
        table["mmd2"][level], table["mmd_p"][level] = mmd2, p
    df = pd.DataFrame(table)
    df.to_csv(out_dir / "levels.csv", index=False)

    # Rows on the forward-gap level grid, so the gap columns are filled, plus the last level.
    step = int(meta["gap_every"])
    shown = sorted(set((np.linspace(0, num_levels - 2, report_levels) / step).round().astype(int)
                       * step) | {num_levels - 1})
    cols = ["level", "sigma_equiv", "rms_ode_mean", "rms_fwd_mean", "nstd_ode_mean",
            "nkurt_ode_mean", "nlagt_ode_mean", "nlagt_fwd_mean", "cosx0_ode_mean", "cosz_mean",
            "straight_mean", "mmd2", "mmd_p", "gap_ode_mean", "gap_fwd_mean", "gap_ratio"]
    view = df.loc[shown, cols].copy()
    # Keep the nearest kept MMD level for rows the stride skipped.
    mmd_idx = np.array(meta["mmd_levels"])
    for r in view.index:
        nearest = int(mmd_idx[np.abs(mmd_idx - r).argmin()])
        view.loc[r, ["mmd2", "mmd_p"]] = df.loc[nearest, ["mmd2", "mmd_p"]].values
    g_all_o, g_all_f = np.nanmean(g_o), np.nanmean(g_f)
    lines = [
        f"# ODE trajectory states vs forward-diffused states: {meta['model_id']}",
        "",
        f"{n_samples} w=1 trajectories, {meta['steps']} steps, MusicCaps captions from row "
        f"{meta['start_index']}, seeds {meta['seed_base']}+row, git {meta['git_sha'][:10]}. "
        f"Forward state at level i: a_i x0 + s_i eps with a fresh eps; effective noise "
        f"(state - a_i x0)/s_i. Gaps on samples with row % {meta['gap_stride']} == 0, at a level "
        f"stride of {meta['gap_every']} (n={int(np.nanmax(table['gap_n']))} samples per level).",
        "",
        f"**One-step gap, pooled over gap levels: forward/ODE = {g_all_f / g_all_o:.3g}x** "
        f"(ODE {g_all_o:.3e}, forward {g_all_f:.3e}; the LoRA-disabled loss in the trainer's units).",
        f"**MMD (ODE states of half the samples vs forward states of the other half, "
        f"{meta['proj_dim']}-d random projection):** "
        f"{int(np.sum(df['mmd_p'] < 0.01))}/{len(mmd_idx)} kept levels at p < 0.01 "
        f"({num_permutations} permutations; smallest attainable p = {1 / (1 + num_permutations):.4f}).",
        "",
        "Null values from the forward draws: nstd 1, nkurt 0, nlagt 0, cosx0 0. `cosz` is the ODE "
        "effective noise against its own starting noise; `straight` is ||x - (a x0 + s z)|| / ||.||.",
        "",
        markdown_table(view),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    logger.info("wrote {} and {}", out_dir / "levels.csv", out_dir / "REPORT.md")


if __name__ == "__main__":
    fire.Fire({"run": run, "aggregate": aggregate})
