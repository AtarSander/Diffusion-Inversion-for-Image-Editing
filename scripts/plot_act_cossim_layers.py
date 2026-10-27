# ABOUTME: Plots per-layer cosine similarity between DDIM- and MW-inverted activations
# ABOUTME: (paired per example) at three diffusion steps, for SA and CA reads.
import json
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch

STEPS = {0: "near noise", 25: "middle", 49: "near image"}
STEP_COLORS = {0: "#557E98", 25: "#668574", 49: "#B8664E"}


def block_spans(layers: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous [lo, hi) spans of layers sharing a UNet block prefix."""
    spans: list[tuple[str, int, int]] = []
    for i, name in enumerate(layers):
        prefix = name.split(".attentions")[0].replace("_blocks", "").replace("_block", "")
        if spans and spans[-1][0] == prefix:
            spans[-1] = (prefix, spans[-1][1], i + 1)
        else:
            spans.append((prefix, i, i + 1))
    return spans


def main(
    features: str = "output/steering/triangles_size/20261026_172253/extract/features.pt",
    output_dir: str = "output/steering/plots",
) -> None:
    """Paired cos(DDIM, MW) activation similarity across layers at three steps."""
    bundle = torch.load(features, map_location="cpu", weights_only=True)
    records = bundle["records"]
    reps = sorted(records[0]["DDIM"]["features"])
    layers = list(records[0]["DDIM"]["features"][reps[0]])
    n_steps = records[0]["DDIM"]["features"][reps[0]][layers[0]].shape[0]
    assert n_steps == 50, n_steps
    print(f"{len(records)} records, reps={reps}, {len(layers)} layers, {n_steps} steps")

    curves: dict[str, dict[int, dict[str, list[float]]]] = {}
    for rep in reps:
        curves[rep] = {}
        for step in STEPS:
            # [examples, layers]: cosine over channels per example, per layer.
            cos = torch.stack(
                [
                    torch.stack(
                        [
                            torch.nn.functional.cosine_similarity(
                                r["DDIM"]["features"][rep][name][step].float(),
                                r["MW"]["features"][rep][name][step].float(),
                                dim=0,
                            )
                            for name in layers
                        ]
                    )
                    for r in records
                ]
            )
            assert cos.shape == (len(records), len(layers)), cos.shape
            curves[rep][step] = {
                "mean": cos.mean(0).tolist(),
                "sem": (cos.std(0) / len(records) ** 0.5).tolist(),
            }
        print(f"{rep}: step0 mean cos {sum(curves[rep][0]['mean']) / len(layers):.4f}")

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(len(reps), 1, figsize=(13, 4.6 * len(reps)), sharex=True)
    x = range(len(layers))
    for ax, rep in zip(axes, reps):
        for step, label in STEPS.items():
            c = curves[rep][step]
            mean = torch.tensor(c["mean"])
            sem = torch.tensor(c["sem"])
            t = 981 - 20 * step
            ax.plot(
                x, mean, color=STEP_COLORS[step], linewidth=2,
                label=f"step {step + 1}/50, t={t} ({label})",
            )
            ax.fill_between(x, mean - sem, mean + sem, color=STEP_COLORS[step], alpha=0.25)
        for name, lo, hi in block_spans(layers):
            if lo:
                ax.axvline(lo - 0.5, color="gray", linewidth=0.8, alpha=0.5)
            ax.text(
                (lo + hi - 1) / 2, 0.02, name, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=14, style="italic", color="gray",
            )
        ax.set_ylabel(f"cos(DDIM, MW) — {rep}")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc="lower left", fontsize=14)
    axes[-1].set_xlabel("transformer block (forward order)")
    dataset = Path(features).parents[2].name
    fig.suptitle(f"DDIM- vs MW-inverted activations, paired per example ({dataset})")
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = out / f"{ts}_act_cossim_layers_{dataset}.png"
    fig.savefig(png, dpi=160)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {
        "features": str(features),
        "git_sha": sha,
        "n_records": len(records),
        "layers": layers,
        "steps": {s: STEPS[s] for s in STEPS},
        "curves": curves,
    }
    (out / f"{ts}_act_cossim_layers_{dataset}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {png}")


if __name__ == "__main__":
    fire.Fire(main)
