# ABOUTME: Re-renders the DDIM-vs-MW activation cosine comparison in the viz house
# ABOUTME: style, from the JSON written by scripts/plot_act_cossim_layers.py.
import json
from pathlib import Path

import viz

PLOTS = Path("output/steering/plots")
INPUTS = {
    "triangles": PLOTS / "20261027_141029_act_cossim_layers.json",
    "ukiyo-e": PLOTS / "20261027_141151_act_cossim_layers_sweep_ukiyo_e.json",
}
TITLES = {
    "triangles": "On triangles, DDIM and MW activations never drop below cos 0.998",
    "ukiyo-e": "On ukiyo-e, activations match at t=1 but reach cos 0.91 mid-trajectory",
}
STEP_NAMES = {"0": "t=981 (near noise)", "25": "t=481 (middle)", "49": "t=1 (near image)"}


def block_spans(layers: list[str]) -> list[tuple[str, int]]:
    """(block name, first layer index) for each contiguous UNet block prefix."""
    spans: list[tuple[str, int]] = []
    for i, name in enumerate(layers):
        prefix = name.split(".attentions")[0].replace("_blocks", "").replace("_block", "")
        if not spans or spans[-1][0] != prefix:
            spans.append((prefix, i))
    return spans


def panel(
    curves: dict, rep: str, marks: dict, letter: str, label: str,
    decimals: int | None, ticks: list[float] | None,
) -> viz.Figure:
    series = {STEP_NAMES[s]: curves[rep][s]["mean"] for s in STEP_NAMES}
    bands = {
        STEP_NAMES[s]: (
            [m - e for m, e in zip(curves[rep][s]["mean"], curves[rep][s]["sem"])],
            [m + e for m, e in zip(curves[rep][s]["mean"], curves[rep][s]["sem"])],
        )
        for s in STEP_NAMES
    }
    return viz.line(
        series, x=list(range(len(curves[rep]["0"]["mean"]))), bands=bands, marks=marks,
        unit="", zero=False, markers=False, decimals=decimals, ticks=ticks,
        xlabel="Transformer block (forward order)",
        ylabel="Cosine similarity, DDIM vs MW", title=f"({letter}) {label}",
        roles={"ordered": list(STEP_NAMES.values())},
    )


def main() -> None:
    for dataset, path in INPUTS.items():
        meta = json.loads(path.read_text())
        marks = {i: name for name, i in block_spans(meta["layers"]) if i}
        decimals = 4 if dataset == "triangles" else None
        ticks = {
            "SA": [0.998, 0.999, 1.0] if dataset == "triangles" else None,
            "CA": [0.999, 1.0] if dataset == "triangles" else None,
        }
        fig = viz.panels(
            [
                panel(meta["curves"], "SA", marks, "a", "Self-attention output",
                      decimals, ticks["SA"]),
                panel(meta["curves"], "CA", marks, "b", "Cross-attention output",
                      decimals, ticks["CA"]),
            ],
            cols=1,
            roles={"ordered": list(STEP_NAMES.values())},
            title=TITLES[dataset],
            subtitle="Bands are ±1 standard-error intervals",
        )
        out = PLOTS / f"{path.stem}_viz.png"
        fig.save(str(out))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
