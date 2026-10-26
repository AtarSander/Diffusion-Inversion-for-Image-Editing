# ABOUTME: Aggregates the CelebA layer-ablation sweep (3 attributes x 10 layer setups):
# ABOUTME: paired dP_probe / id_sim / found-rate vs alpha per setup, plus MW-vs-DDIM split.
import glob
import json
import math
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
ATTRS = {"smiling": "Smiling", "eyeglasses": "Eyeglasses", "young": "Young"}
SETUPS = [
    "mid", "up0", "up1", "up0a0", "up0a1", "up0a01", "up0a2",
    "up1a0", "up1a1", "up1a2",
]
COLORS = {
    "last36": "#6E6C65", "mid": "#191918",
    "up0": "#2F5E78", "up0a0": "#557E98", "up0a1": "#7FA3B8", "up0a01": "#3E7FA6",
    "up0a2": "#A8C4D4",
    "up1": "#94503A", "up1a0": "#B8664E", "up1a1": "#CC8A74", "up1a2": "#E0B0A0",
}
OUT = ROOT / "output/steering/celeba_summary"
TS = time.strftime("%Y%m%d_%H%M%S")


def sem(values):
    """Standard error of the mean (0 for n<=1)."""
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def load_rows(attr, setup):
    """Return scored image rows for one (attribute, layer-setup) run."""
    pattern = (
        f"output/steering/celeba_{attr}/*/scores_faces_*/*_face_scores.json"
        if setup == "last36"
        else f"output/steering/celeba_layers/{attr}_{setup}/*/scores_faces_*/*_face_scores.json"
    )
    hits = sorted(glob.glob(str(ROOT / pattern)))
    assert hits, f"no scores for {attr}/{setup}"
    return json.loads(Path(hits[-1]).read_text())["images"]


def curves(rows, methods=("DDIM", "MW")):
    """Per alpha: paired deltas vs base (p_probe, id_sim) and face-found rate."""
    base = {(r["prompt"], r["seed"]): r for r in rows if r["method"] == "base"}
    steered = [r for r in rows if r["method"] in methods]
    out = {}
    for a in sorted({r["strength"] for r in steered}):
        sub = [r for r in steered if r["strength"] == a]
        rec = {"found": sum(not math.isnan(r["p_probe"]) for r in sub) / len(sub)}
        for metric in ("p_probe", "id_sim"):
            deltas = [
                r[metric] - base[(r["prompt"], r["seed"])][metric]
                for r in sub
                if not math.isnan(r[metric])
                and not math.isnan(base[(r["prompt"], r["seed"])][metric])
            ]
            rec[metric] = (
                (statistics.mean(deltas), sem(deltas), len(deltas))
                if deltas
                else (float("nan"), 0.0, 0)
            )
        out[a] = rec
    return out


def fmt(cur, a, metric, plus=0.0):
    """Format mean±sem at one alpha; dash when the run's grid lacks that alpha."""
    if a not in cur:
        return "–"
    m, s, _ = cur[a][metric]
    return f"{m + plus:+.3f}±{s:.3f}" if not math.isnan(m) else "nan"


def main():
    lines = [f"# CelebA layer ablation ({TS})", "",
             "Paired deltas vs base per (prompt, seed); MW+DDIM pooled unless noted.",
             "last36 = main-run reference (up_blocks.0 + up_blocks.1)."]
    for attr, attr_name in ATTRS.items():
        data = {s: curves(load_rows(attr, s)) for s in ["last36", *SETUPS]}

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        titles = [f"ΔP_probe({attr_name}) vs base",
                  "Δ id_sim vs base (+1 = base identity)", "faces found"]
        for setup, cur in data.items():
            alphas = sorted(cur)
            style = dict(
                color=COLORS[setup],
                lw=2.4 if setup in ("last36", "mid", "up0", "up1") else 1.4,
                ls=(0, (5, 3)) if setup == "last36" else "-", marker="o", ms=3.5,
                label=setup + (" (ref)" if setup == "last36" else ""),
            )
            axes[0].errorbar(alphas, [cur[a]["p_probe"][0] for a in alphas],
                             yerr=[cur[a]["p_probe"][1] for a in alphas], capsize=2, **style)
            axes[1].errorbar(alphas, [cur[a]["id_sim"][0] + 1 for a in alphas],
                             yerr=[cur[a]["id_sim"][1] for a in alphas], capsize=2, **style)
            axes[2].plot(alphas, [cur[a]["found"] for a in alphas], **style)
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("alpha")
            ax.grid(alpha=0.25)
        axes[0].axhline(0, color="#B9B6AE", lw=0.8, ls=":")
        axes[1].axhline(1.0, color="#B9B6AE", lw=0.8, ls=":")
        axes[1].set_ylim(bottom=0)
        axes[2].set_ylim(0, 1.05)
        axes[2].legend(fontsize=8, ncol=2, loc="lower left")
        fig.suptitle(f"{attr_name} layer ablation (MW+DDIM pooled; last36 = dashed reference)", y=1.0)
        fig.tight_layout()
        png = OUT / f"{TS}_{attr}_layers.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")

        lines += ["", f"## {attr_name}", "",
                  "| setup | dP_probe a=+2 | dP_probe a=+3 | dP_probe a=-1 | dP_probe a=-1.5 "
                  "| id_sim a=+2 | id_sim a=+3 | found a=+3 |",
                  "|---|---|---|---|---|---|---|---|"]
        for setup, cur in data.items():
            lines.append(
                f"| {setup} | {fmt(cur, 2.0, 'p_probe')} | {fmt(cur, 3.0, 'p_probe')}"
                f" | {fmt(cur, -1.0, 'p_probe')} | {fmt(cur, -1.5, 'p_probe')}"
                f" | {fmt(cur, 2.0, 'id_sim', 1)} | {fmt(cur, 3.0, 'id_sim', 1)}"
                f" | {cur[3.0]['found']:.0%} |"
            )

        lines += ["", f"### {attr_name}: MW vs DDIM, dP_probe", "",
                  "| setup | DDIM a=+3 | MW a=+3 | DDIM a=-1 | MW a=-1 |",
                  "|---|---|---|---|---|"]
        for setup in ["last36", *SETUPS]:
            ddim = curves(load_rows(attr, setup), methods=("DDIM",))
            mw = curves(load_rows(attr, setup), methods=("MW",))
            lines.append(
                f"| {setup} | {fmt(ddim, 3.0, 'p_probe')} | {fmt(mw, 3.0, 'p_probe')}"
                f" | {fmt(ddim, -1.0, 'p_probe')} | {fmt(mw, -1.0, 'p_probe')} |"
            )

    md = OUT / f"{TS}_celeba_layers_summary.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
