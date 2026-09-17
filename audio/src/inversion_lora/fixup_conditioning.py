# ABOUTME: Detach conditioning tensors in an existing trajectory/real-pairs dataset in place, so
# ABOUTME: the DataLoader can serialize them (fixes non-leaf-requires-grad save from a generator).

import sys
from pathlib import Path

import fire
import torch
from tqdm import tqdm


def main(root_dir: str) -> None:
    """Rewrite every sample's conditioning.pt with detached, grad-free tensors.

    Args:
        root_dir: Dataset directory containing `sample_*` subdirectories.
    """
    root = Path(root_dir)
    samples = sorted(root.glob("sample_*"))
    if not samples:
        raise FileNotFoundError(f"no sample_* under {root}")
    fixed = 0
    for sample_dir in tqdm(samples, desc="detaching conditioning"):
        path = sample_dir / "conditioning.pt"
        cond = torch.load(path, map_location="cpu", weights_only=True)
        needs = any(getattr(v, "requires_grad", False) for v in cond.values())
        if needs:
            torch.save({k: v.detach().clone() for k, v in cond.items()}, path)
            fixed += 1
    print(f"{len(samples)} samples, {fixed} rewritten (rest already grad-free)")
    if fixed == 0:
        print("nothing needed fixing", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(main)
