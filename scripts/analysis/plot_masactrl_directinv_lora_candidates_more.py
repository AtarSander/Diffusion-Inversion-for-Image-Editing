#!/usr/bin/env python3
"""Render a second non-overlapping set of MasaCtrl qualitative candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .plot_masactrl_directinv_lora_candidates import RESULT_DIR
    from .plot_pnp_directinv_lora_candidates import Candidate, render
except ImportError:
    from plot_masactrl_directinv_lora_candidates import RESULT_DIR
    from plot_pnp_directinv_lora_candidates import Candidate, render


CANDIDATES = (
    Candidate("M", "311000000002", "Remove the laptop"),
    Candidate("N", "312000000004", "Remove the campfire"),
    Candidate("O", "324000000000", "Remove the pink lotus"),
    Candidate("P", "000000000081", "Remove the mushrooms"),
    Candidate("Q", "324000000009", "Remove the church"),
    Candidate("R", "000000000110", "Make the woman older"),
    Candidate("S", "712000000001", "Man → robot"),
    Candidate("T", "722000000001", "Man → sculpture"),
    Candidate("U", "112000000000", "Dress → shirt"),
    Candidate("V", "314000000007", "Remove the dog from the painting"),
    Candidate("W", "312000000003", "Remove the bat"),
    Candidate("X", "000000000041", "Add a castle in the aurora"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/PIE-Bench_v1")
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/masactrl_directinv_lora_candidates_more.png"),
    )
    args = parser.parse_args()
    render(
        args.data_root,
        args.evaluation_root,
        args.output,
        candidates=CANDIDATES,
        editor_label="MasaCtrl",
        editor_dir="masactrl",
        result_dir=RESULT_DIR,
    )


if __name__ == "__main__":
    main()
