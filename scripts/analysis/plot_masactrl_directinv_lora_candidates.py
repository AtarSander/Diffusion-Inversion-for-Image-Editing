#!/usr/bin/env python3
"""Render visually screened MasaCtrl qualitative candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .plot_pnp_directinv_lora_candidates import Candidate, render
except ImportError:
    from plot_pnp_directinv_lora_candidates import Candidate, render


CANDIDATES = (
    Candidate("A", "213000000000", "Add hearts to the background"),
    Candidate("B", "121000000002", "Horse → zebra"),
    Candidate("C", "124000000003", "Tiger → dog"),
    Candidate("D", "000000000111", "Cat → dog"),
    Candidate("E", "111000000007", "Cat → bear"),
    Candidate("F", "111000000003", "Bunny → pig"),
    Candidate("G", "211000000001", "Add a gold chain and star"),
    Candidate("H", "000000000076", "Smoke → flower"),
    Candidate("I", "000000000006", "Tulip → lion latte art"),
    Candidate("J", "000000000134", "Bird → red and white"),
    Candidate("K", "712000000004", "Garage-kit style"),
    Candidate("L", "712000000002", "Woman → golden sculpture"),
)

RESULT_DIR = "one_epoch/cycle_loss_cfg1_r16_lambda4_uniform_e1_all_editors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/PIE-Bench_v1")
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/masactrl_directinv_lora_candidates.png"),
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
