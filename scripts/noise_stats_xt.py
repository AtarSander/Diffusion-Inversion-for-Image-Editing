# ABOUTME: Compares DDIM vs MW inverted x_T latents against true Gaussian noise:
# ABOUTME: moments, spatial autocorrelation, leak from x_0, radial power spectrum.
"""Noise statistics of inverted x_T latents (DDIM vs MW vs Gaussian baseline).

Reads saved inversion trajectories from extract dirs
(`inversions/source_XXX_{DDIM,MW}.pt`, states[0]=x_0 clean latent,
states[-1]=x_T) and reports, per method:

- element moments: mean, std, skewness, excess kurtosis
- lag-1 spatial autocorrelation (mean of horizontal and vertical)
- Pearson correlation of x_T with x_0 (structure leak)
- per-channel mean bias
- radially averaged power spectrum (white noise = flat at 1)

Usage:
    uv run python scripts/noise_stats_xt.py \
        --runs '{"ukiyo_e": "output/steering/sweep_ukiyo_e/20261024_143816/extract"}' \
        [--limit 4]
"""

import json
import subprocess
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

METHODS = ("DDIM", "MW", "Gaussian")
DEFAULT_RUNS = {
    "ukiyo_e": "output/steering/sweep_ukiyo_e/20261024_143816/extract",
    "triangles": "output/steering/triangles_size/20261026_172253/extract",
}
COLORS = {"DDIM": "#d62728", "MW": "#1f77b4", "Gaussian": "#7f7f7f"}

plt.rcParams.update({"font.size": 14})


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two equally shaped tensors."""
    assert a.shape == b.shape, (a.shape, b.shape)
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def latent_stats(x_t: torch.Tensor, x_0: torch.Tensor) -> dict[str, float]:
    """Per-image noise statistics of one x_T latent [1, 4, H, W]."""
    assert x_t.ndim == 4 and x_t.shape[:2] == (1, 4), x_t.shape
    z = (x_t - x_t.mean()) / x_t.std()
    lag_h = pearson(x_t[..., :-1], x_t[..., 1:])
    lag_v = pearson(x_t[..., :-1, :], x_t[..., 1:, :])
    center = lambda t: t - t.mean(dim=(0, 2, 3), keepdim=True)  # noqa: E731
    return {
        "mean": float(x_t.mean()),
        "std": float(x_t.std()),
        "skew": float((z**3).mean()),
        "kurtosis": float((z**4).mean() - 3.0),
        "lag1_autocorr": (lag_h + lag_v) / 2.0,
        "corr_x0": pearson(center(x_t), center(x_0)),
        "channel_means": x_t.mean(dim=(0, 2, 3)).tolist(),
    }


def radial_spectrum(x_t: torch.Tensor) -> np.ndarray:
    """Radially averaged power spectrum, normalized so white noise is flat at 1."""
    assert x_t.ndim == 4, x_t.shape
    power = torch.fft.fftshift(torch.fft.fft2(x_t[0]).abs() ** 2, dim=(-2, -1))
    h, w = power.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing="ij"
    )
    radius = (yy**2 + xx**2).sqrt().round().long()
    n_bins = h // 2
    spectrum = np.zeros(n_bins)
    flat_r, flat_p = radius.flatten(), power.mean(dim=0).flatten()
    for r in range(n_bins):
        spectrum[r] = float(flat_p[flat_r == r].mean())
    return spectrum / power.mean().item()


def load_xt_x0(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (x_T, x_0) in float32 from one saved inversion trajectory."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    states = data["states"]
    assert data["state_timesteps"][0] is None, "states[0] must be the clean latent"
    return states[-1].float(), states[0].float()


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std across images for each scalar statistic."""
    keys = [k for k in rows[0] if k != "channel_means"]
    out = {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows])),
        }
        for k in keys
    }
    out["channel_means"] = {
        "mean": np.mean([r["channel_means"] for r in rows], axis=0).tolist(),
        "std": np.std([r["channel_means"] for r in rows], axis=0).tolist(),
    }
    return out


def plot_spectra(spectra: dict, out_png: Path) -> None:
    """One panel per dataset: mean radial power spectrum per method."""
    datasets = list(spectra)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, datasets):
        for method in METHODS:
            mean = np.mean(spectra[name][method], axis=0)
            std = np.std(spectra[name][method], axis=0)
            freq = np.arange(len(mean))
            ax.plot(freq, mean, label=method, color=COLORS[method], linewidth=2)
            ax.fill_between(
                freq, mean - std, mean + std, color=COLORS[method], alpha=0.15
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("radial frequency")
        ax.set_ylabel("normalized power")
        ax.set_title(f"x_T power spectrum — {name}")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_examples(
    examples: dict[str, list[dict]], out_png: Path, per_dataset: int
) -> None:
    """Grid: source image | DDIM x_T | MW x_T | Gaussian (channel-mean, z-scored)."""
    rows = [(name, ex) for name, items in examples.items() for ex in items]
    cols = ["source", "DDIM x_T", "MW x_T", "Gaussian"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (name, ex) in enumerate(rows):
        axes[i, 0].imshow(plt.imread(ex["source"]))
        axes[i, 0].set_ylabel(name, fontsize=14)
        for j, key in enumerate(("DDIM", "MW", "Gaussian"), start=1):
            m = ex[key].mean(axis=0)
            z = (m - m.mean()) / m.std()
            axes[i, j].imshow(z, cmap="gray", vmin=-3, vmax=3)
        for j, title in enumerate(cols):
            if i == 0:
                axes[i, j].set_title(title, fontsize=14)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def markdown_report(results: dict, paths: dict[str, str]) -> str:
    """Render the aggregate tables as markdown."""
    lines = [
        "# x_T noise statistics: DDIM vs MW vs Gaussian",
        "",
        "x_T = states[-1] of saved inversion trajectories; Gaussian = seeded "
        "randn of the same shape. lag1 = mean lag-1 spatial autocorrelation; "
        "corr_x0 = Pearson corr with the clean latent after removing per-channel "
        "means (spatial structure leak). "
        "Values are mean ± std across images.",
        "",
    ]
    for name, run in paths.items():
        lines += [f"## {name}", "", f"Run: `{run}`", ""]
        lines.append(
            "| method | mean | std | skew | kurtosis | lag1 autocorr | corr_x0 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for method in METHODS:
            agg = results[name][method]
            cells = [
                f"{agg[k]['mean']:+.4f}±{agg[k]['std']:.4f}"
                for k in ("mean", "std", "skew", "kurtosis", "lag1_autocorr", "corr_x0")
            ]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        channel_rows = [
            f"| {method} | "
            + " | ".join(f"{v:+.4f}" for v in results[name][method]["channel_means"]["mean"])
            + " |"
            for method in METHODS
        ]
        lines += [
            "",
            "Per-channel mean of x_T:",
            "",
            "| method | ch0 | ch1 | ch2 | ch3 |",
            "|---|---|---|---|---|",
            *channel_rows,
            "",
        ]
    return "\n".join(lines) + "\n"


def main(
    runs: dict[str, str] | None = None,
    output_dir: str = "output/steering/plots",
    limit: int | None = None,
    examples_per_dataset: int = 2,
) -> None:
    """Compute x_T noise statistics for each extract run and write md/json/png.

    Args:
        runs: dataset name -> extract dir containing `inversions/` and
            `sources.json`. Defaults to the ukiyo-e sweep and triangles-size.
        output_dir: where the timestamped report files go.
        limit: only analyze the first N sources (smoke run).
        examples_per_dataset: rows in the x_T visualization per dataset.
    """
    runs = runs or DEFAULT_RUNS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    results: dict = {}
    spectra: dict = {}
    examples: dict = {}
    generator = torch.Generator().manual_seed(0)
    for name, run in runs.items():
        sources = json.loads((Path(run) / "sources.json").read_text())[:limit]
        stats = {m: [] for m in METHODS}
        spectra[name] = {m: [] for m in METHODS}
        examples[name] = []
        for index, source in enumerate(sources):
            per_method: dict[str, np.ndarray] = {}
            for method in ("DDIM", "MW"):
                x_t, x_0 = load_xt_x0(
                    Path(run) / "inversions" / f"source_{index:03d}_{method}.pt"
                )
                stats[method].append(latent_stats(x_t, x_0))
                spectra[name][method].append(radial_spectrum(x_t))
                per_method[method] = x_t[0].numpy()
            gauss = torch.randn(x_t.shape, generator=generator)
            stats["Gaussian"].append(latent_stats(gauss, x_0))
            spectra[name]["Gaussian"].append(radial_spectrum(gauss))
            per_method["Gaussian"] = gauss[0].numpy()
            if index < examples_per_dataset:
                examples[name].append({"source": source["path"], **per_method})
            if index == 0:
                print(f"[{name}] first x_T stats (DDIM): {stats['DDIM'][0]}")
            print(f"[{name}] {index + 1}/{len(sources)}", flush=True)
        results[name] = {m: aggregate(stats[m]) for m in METHODS}

    spectrum_png = out / f"{ts}_xt_spectrum.png"
    examples_png = out / f"{ts}_xt_examples.png"
    plot_spectra(spectra, spectrum_png)
    plot_examples(examples, examples_png, examples_per_dataset)
    report = markdown_report(results, runs)
    (out / f"{ts}_xt_noise_stats.md").write_text(report)
    payload = {
        "git_sha": sha,
        "timestamp": ts,
        "runs": runs,
        "limit": limit,
        "results": results,
        "mean_spectra": {
            name: {m: np.mean(v, axis=0).tolist() for m, v in by.items()}
            for name, by in spectra.items()
        },
    }
    (out / f"{ts}_xt_noise_stats.json").write_text(json.dumps(payload, indent=2))
    print(report)
    print(f"wrote {out / f'{ts}_xt_noise_stats.md'}")
    print(f"wrote {spectrum_png}")
    print(f"wrote {examples_png}")


if __name__ == "__main__":
    fire.Fire(main)
