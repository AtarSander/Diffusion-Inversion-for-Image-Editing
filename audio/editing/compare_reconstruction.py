# ABOUTME: Compare DDIM reconstruction of real MedleyDB audio with and without an inversion LoRA,
# ABOUTME: paired over the same tracks so a small consistent effect is still resolvable at n=35.

from pathlib import Path

import fire
import pandas as pd
from scipy import stats

from editing.run_metrics import per_example

METRICS = [
    ("LPAPS", "lpaps", "lower"),
    ("mel PSNR", "psnr", "higher"),
    ("mel SSIM", "ssim", "higher"),
]


def load(run: Path, column: str) -> pd.Series | None:
    """Per-example metric, or None if that metric was not written for this run."""
    try:
        return per_example(run, column)
    except (FileNotFoundError, KeyError):
        return None


def main(root: str, steps: str = "200,50", baseline: str = "nolora", variants: str = "attn,full"):
    """Print paired comparisons of each LoRA variant against the no-LoRA reconstruction.

    Args:
        root: Directory holding the `recon_tracks_s{steps}_{variant}` runs.
        steps: Comma-separated step counts to report.
        baseline: Variant name treated as the reference.
        variants: Comma-separated variants to compare against it.
    """
    root = Path(root)
    step_list = [s.strip() for s in str(steps).split(",")]
    variant_list = [v.strip() for v in str(variants).split(",")]

    for step in step_list:
        base_run = root / f"recon_tracks_s{step}_{baseline}"
        print(f"\n{'=' * 78}\n{step}-step DDIM reconstruction, 35 real MedleyDB tracks\n{'=' * 78}")
        if not base_run.exists():
            print(f"  missing: {base_run}")
            continue

        print(f"\n{'metric':10s} {'no LoRA':>18} " + " ".join(f"{v + ' LoRA':>18}" for v in variant_list))
        rows = {}
        for label, column, _ in METRICS:
            base = load(base_run, column)
            if base is None:
                print(f"{label:10s} {'not scored yet':>18}")
                continue
            cells = [f"{base.mean():.3f} ± {base.sem():.3f}"]
            rows[label] = (base, {})
            for variant in variant_list:
                other = load(root / f"recon_tracks_s{step}_{variant}", column)
                if other is None:
                    cells.append(f"{'-':>18}")
                    continue
                rows[label][1][variant] = other
                cells.append(f"{other.mean():.3f} ± {other.sem():.3f}")
            print(f"{label:10s} " + " ".join(f"{c:>18}" for c in cells))

        for label, (base, others) in rows.items():
            better = next(d for m, _, d in METRICS if m == label)
            for variant, other in others.items():
                common = base.index.intersection(other.index)
                delta = other.loc[common] - base.loc[common]
                t, p = stats.ttest_rel(other.loc[common], base.loc[common])
                wins = int((delta > 0).sum()) if better == "higher" else int((delta < 0).sum())
                direction = "better" if (
                    (delta.mean() > 0) == (better == "higher")
                ) else "WORSE"
                ci = 1.96 * delta.std() / len(delta) ** 0.5
                print(
                    f"  {label:9s} {variant:5s} - no LoRA = {delta.mean():+8.4f} "
                    f"[{delta.mean() - ci:+.4f}, {delta.mean() + ci:+.4f}]  p={p:.4f}  "
                    f"{wins}/{len(delta)} tracks {direction}"
                )


if __name__ == "__main__":
    fire.Fire(main)
