# ABOUTME: Compares two fitted steering-vector files (directions.pt): per-layer cosine
# ABOUTME: between A and B per method, plus norm ratios and B's DDIM-vs-MW agreement.
import json
import subprocess
import time
from pathlib import Path

import fire
import torch


def load(path: str) -> dict:
    """Load directions.pt and return the nested directions dict."""
    return torch.load(path, map_location="cpu", weights_only=True)["directions"]


def pair_stats(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Mean cosine over timesteps and mean norm ratio ||y||/||x||."""
    assert x.shape == y.shape and x.ndim == 2, (x.shape, y.shape)
    cos = torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=1)
    ratio = y.float().norm(dim=1) / x.float().norm(dim=1).clamp(min=1e-12)
    return cos.mean().item(), ratio.mean().item()


def main(
    a: str,
    b: str,
    name_a: str = "filled",
    name_b: str = "outline",
    output_dir: str = "output/steering/plots",
) -> None:
    """Write a timestamped md+json comparison of two directions.pt files."""
    da, db = load(a), load(b)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    results: dict = {"a": a, "b": b, "git_sha": sha, "reps": {}}
    lines = [
        f"# SV comparison: {name_a} (A) vs {name_b} (B)",
        "",
        f"A: `{a}`",
        f"B: `{b}`",
        "",
        "cos = mean over 50 timesteps of per-layer cosine; norm = mean ||B||/||A||.",
        "",
    ]
    for rep in sorted(set(da) & set(db)):
        results["reps"][rep] = {}
        lines += [
            f"## {rep}",
            "",
            "| method | mean cos A,B | min..max layer cos | mean norm B/A |",
            "|---|---|---|---|",
        ]
        for method in sorted(set(da[rep]) & set(db[rep])):
            la, lb = da[rep][method], db[rep][method]
            assert set(la) == set(lb), (rep, method)
            per_layer = {}
            for layer in la:
                cos, ratio = pair_stats(la[layer]["true"], lb[layer]["true"])
                per_layer[layer] = {"cos": cos, "norm_ratio": ratio}
            cs = [v["cos"] for v in per_layer.values()]
            rs = [v["norm_ratio"] for v in per_layer.values()]
            results["reps"][rep][method] = per_layer
            lines.append(
                f"| {method} | {sum(cs) / len(cs):.4f} | "
                f"{min(cs):.4f}..{max(cs):.4f} | {sum(rs) / len(rs):.3f} |"
            )
        # agreement of the two inversion methods WITHIN each file
        for tag, d in ((name_a, da), (name_b, db)):
            cs = [
                pair_stats(d[rep]["DDIM"][layer]["true"], d[rep]["MW"][layer]["true"])[0]
                for layer in d[rep]["DDIM"]
            ]
            lines.append(
                f"| DDIM vs MW ({tag}) | {sum(cs) / len(cs):.4f} | {min(cs):.4f}..{max(cs):.4f} | - |"
            )
        lines.append("")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{ts}_sv_compare_{name_a}_vs_{name_b}"
    stem.with_suffix(".json").write_text(json.dumps(results, indent=1) + "\n")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {stem}.md and .json")


if __name__ == "__main__":
    fire.Fire(main)
