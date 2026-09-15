# ABOUTME: Strips the leaked unconditional keys out of a pair-branch conditional checkpoint, so a
# ABOUTME: run trained before the adapter-name guard is usable without retraining.

from pathlib import Path

import fire
import torch
from loguru import logger

MARKER = "_uncond."


def main(checkpoint_dir: str, dry_run: bool = False) -> None:
    """Rewrite every `checkpoint*.pt` whose conditional file also holds the other adapter.

    PEFT filters state dicts by adapter name with string matching, so naming the second adapter
    `inversion_uncond` next to `inversion` wrote both into the conditional file (768 keys instead
    of 384). The unconditional sidecar was always correct. Keys are identified by the `lora_*_uncond.`
    module suffix PEFT gives the second adapter, and the originals are kept as `.pt.bak`.

    Args:
        checkpoint_dir: Directory holding the pair-branch checkpoints.
        dry_run: Report what would change without writing.
    """
    root = Path(checkpoint_dir)
    targets = [p for p in sorted(root.glob("*.pt")) if not p.stem.endswith("_uncond")]
    if not targets:
        raise FileNotFoundError(f"no conditional checkpoints under {root}")

    for path in targets:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            logger.info("{}: not a state dict, skipped", path.name)
            continue
        leaked = [k for k in state if MARKER in k]
        if not leaked:
            logger.info("{}: clean ({} keys)", path.name, len(state))
            continue
        cleaned = {k: v for k, v in state.items() if MARKER not in k}
        logger.warning(
            "{}: {} keys -> {} ({} leaked from the other adapter)",
            path.name, len(state), len(cleaned), len(leaked),
        )
        assert cleaned, f"{path}: stripping left nothing, the marker {MARKER!r} is wrong"
        if dry_run:
            continue
        path.replace(path.with_suffix(".pt.bak"))
        torch.save(cleaned, path)
        logger.success("{}: rewritten, original kept as {}", path.name, path.name + ".bak")


if __name__ == "__main__":
    fire.Fire(main)
