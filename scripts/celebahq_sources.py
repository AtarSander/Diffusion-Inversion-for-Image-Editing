# ABOUTME: Selects covariate-matched positive/negative CelebA-HQ faces per target
# ABOUTME: attribute and writes steering source sets in the wikiart32 layout.
"""Balanced CelebA-HQ steering sources.

For each target attribute, draws `count` positive/negative pairs exactly matched
on the remaining covariates (Male, Young, Smiling, Eyeglasses minus the target),
so the mean-difference direction cannot encode those confounds. The kth positive
pairs with the kth negative, which the fit's swap control relies on.

Run after unzipping CelebAMask-HQ.zip under data/raw/celebamask_hq:
    uv run python scripts/celebahq_sources.py
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import fire

from diff_inversion.steering.face_probe import CELEBAHQ_ROOT, IMG_DIR, load_anno

BALANCE = ("Male", "Young", "Smiling", "Eyeglasses")


def matched_pairs(
    rows: list[dict], target: str, count: int, seed: int
) -> tuple[list[tuple[dict, dict]], list[str]]:
    """Greedy exact matching of positives to negatives on the non-target covariates."""
    covariates = [c for c in BALANCE if c != target]

    def cell(row: dict) -> tuple[int, ...]:
        return tuple(row[c] for c in covariates)

    positives = [r for r in rows if r[target] == 1]
    negatives = defaultdict(list)
    for row in rows:
        if row[target] == -1:
            negatives[cell(row)].append(row)
    rng = random.Random(seed)
    rng.shuffle(positives)
    for bucket in negatives.values():
        rng.shuffle(bucket)
    pairs = []
    for positive in positives:
        bucket = negatives[cell(positive)]
        if bucket:
            pairs.append((positive, bucket.pop()))
        if len(pairs) == count:
            break
    if len(pairs) != count:
        raise RuntimeError(f"Only {len(pairs)} matched pairs for {target}")
    return pairs, covariates


def main(
    targets: tuple[str, ...] = ("Smiling", "Eyeglasses", "Young"),
    count: int = 32,
    seed: int = 42,
    out_root: str = "data/steering",
) -> None:
    """Write celebahq32_<target>/ dirs with images, sources.jsonl and selection meta."""
    _, rows = load_anno()
    for target in targets:
        pairs, covariates = matched_pairs(rows, target, count, seed)
        out = Path(out_root) / f"celebahq32_{target.lower()}"
        out.mkdir(parents=True, exist_ok=False)
        manifest, selection = [], []
        cells = Counter()
        for index, (positive, negative) in enumerate(pairs):
            for label, row in (("positive", positive), ("negative", negative)):
                name = f"{label}_{index:03d}_{target}.jpg"
                shutil.copy(CELEBAHQ_ROOT / IMG_DIR / row["file"], out / name)
                manifest.append({"id": name[:-4], "path": name, "label": label})
            cells[tuple(positive[c] for c in covariates)] += 1
            selection.append(
                {
                    "pair": index,
                    "positive_file": positive["file"],
                    "negative_file": negative["file"],
                    "cell": dict(zip(covariates, (positive[c] for c in covariates))),
                }
            )
        (out / "sources.jsonl").write_text(
            "\n".join(json.dumps(row) for row in manifest) + "\n"
        )
        (out / "selection.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "covariates": covariates,
                    "count": count,
                    "seed": seed,
                    "pairs": selection,
                },
                indent=2,
            )
        )
        print(f"{target}: {count} pairs; covariate cells (pos == neg by construction):")
        for value, cell_count in sorted(cells.items()):
            print(f"  {dict(zip(covariates, value))}: {cell_count}")


if __name__ == "__main__":
    fire.Fire(main)
