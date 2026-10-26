# ABOUTME: Cross-attribute analysis of the CelebA face-steering runs: alpha curves
# ABOUTME: (dP_probe, dP_clip, id_sim, face-found rate), MW-vs-DDIM paired deltas, off-target drift.
import json
import math
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
RUNS = {
    "Smiling": ROOT
    / "output/steering/celeba_smiling/20261026_024550/scores_faces_20261026_055223/20261026_055223_face_scores.json",
    "Eyeglasses": ROOT
    / "output/steering/celeba_eyeglasses/20261026_024550/scores_faces_20261026_045032/20261026_045032_face_scores.json",
    "Young": ROOT
    / "output/steering/celeba_young/20261026_024550/scores_faces_20261026_045419/20261026_045419_face_scores.json",
}
OUT = ROOT / "output/steering/celeba_summary"
OUT.mkdir(exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")
COLORS = {"MW": "#B8664E", "DDIM": "#557E98"}
OFFTARGET = ("Male", "Young", "Smiling", "Eyeglasses")


def sem(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def curves(rows):
    """Per (method, alpha): paired deltas vs base and face-found rate."""
    base = {(r["prompt"], r["seed"]): r for r in rows if r["method"] == "base"}
    out = {}
    for method in ("DDIM", "MW"):
        members = [r for r in rows if r["method"] == method]
        alphas = sorted({r["strength"] for r in members})
        for a in alphas:
            sub = [r for r in members if r["strength"] == a]
            rec = {"found": sum(not math.isnan(r["p_probe"]) for r in sub) / len(sub)}
            for metric in ("p_probe", "p_target", "clip_t", "id_sim"):
                deltas = [
                    r[metric] - base[(r["prompt"], r["seed"])][metric]
                    for r in sub
                    if not math.isnan(r[metric])
                    and not math.isnan(base[(r["prompt"], r["seed"])][metric])
                ]
                rec[f"d_{metric}"] = (
                    (statistics.mean(deltas), sem(deltas), len(deltas))
                    if deltas
                    else (float("nan"), 0.0, 0)
                )
            out[(method, a)] = rec
    return out


def paired_mw_ddim(rows, metric):
    """MW minus DDIM on the same (prompt, alpha), split by alpha sign."""
    by_key = {
        (r["prompt"], r["strength"], r["method"]): r[metric]
        for r in rows
        if r["method"] in ("DDIM", "MW")
    }
    result = {}
    for sign, name in ((1, "pos"), (-1, "neg")):
        deltas = []
        for (prompt, a, method), v in by_key.items():
            if method != "MW" or a * sign <= 0:
                continue
            other = by_key.get((prompt, a, "DDIM"), float("nan"))
            if not math.isnan(v) and not math.isnan(other):
                deltas.append(v - other)
        result[name] = (
            (statistics.mean(deltas), sem(deltas), len(deltas))
            if deltas
            else (float("nan"), 0.0, 0)
        )
    return result


def offtarget_drift(rows, target, alpha=2.0):
    """Mean off-target probe-prob delta vs base at |alpha|=alpha, pooled over methods."""
    base = {(r["prompt"], r["seed"]): r for r in rows if r["method"] == "base"}
    drift = {}
    for attr in OFFTARGET:
        deltas = []
        for r in rows:
            if r["method"] == "base" or abs(r["strength"]) != alpha or r["strength"] < 0:
                continue
            b = base[(r["prompt"], r["seed"])]
            if r.get("probe_attrs") and b.get("probe_attrs"):
                deltas.append(r["probe_attrs"][attr] - b["probe_attrs"][attr])
        drift[attr] = (
            (statistics.mean(deltas), sem(deltas), len(deltas))
            if deltas
            else (float("nan"), 0.0, 0)
        )
    return drift


def fmt(tup):
    m, s, n = tup
    return f"{m:+.3f}±{s:.3f} (n={n})" if not math.isnan(m) else "nan"


def main():
    data = {attr: json.loads(path.read_text()) for attr, path in RUNS.items()}
    all_curves = {attr: curves(d["images"]) for attr, d in data.items()}

    panels = [
        ("d_p_probe", "ΔP_probe vs base"),
        ("d_p_target", "ΔP_clip vs base"),
        ("id_sim", "id_sim to base"),
        ("found", "faces found"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    for col, attr in enumerate(RUNS):
        cur = all_curves[attr]
        for row, (key, label) in enumerate(panels):
            ax = axes[row][col]
            for method in ("DDIM", "MW"):
                alphas = sorted(a for m, a in cur if m == method)
                if key == "found":
                    ys = [cur[(method, a)]["found"] for a in alphas]
                    ax.plot(alphas, ys, "o-", color=COLORS[method], label=method, ms=4)
                elif key == "id_sim":
                    ys = [cur[(method, a)]["d_id_sim"][0] + 1.0 for a in alphas]
                    es = [cur[(method, a)]["d_id_sim"][1] for a in alphas]
                    ax.errorbar(
                        alphas,
                        ys,
                        yerr=es,
                        fmt="o-",
                        color=COLORS[method],
                        label=method,
                        ms=4,
                        capsize=2,
                    )
                else:
                    ys = [cur[(method, a)][key][0] for a in alphas]
                    es = [cur[(method, a)][key][1] for a in alphas]
                    ax.errorbar(
                        alphas,
                        ys,
                        yerr=es,
                        fmt="o-",
                        color=COLORS[method],
                        label=method,
                        ms=4,
                        capsize=2,
                    )
            ax.axhline(
                0 if key.startswith("d_") else (1.0 if key == "id_sim" else None) or 0,
                color="#B9B6AE",
                lw=0.8,
                ls=":",
            )
            if key == "found":
                ax.set_ylim(0, 1.05)
            if row == 0:
                ax.set_title(attr)
            if col == 0:
                ax.set_ylabel(label)
            if row == 3:
                ax.set_xlabel("alpha")
            ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=9)
    fig.suptitle(
        "CelebA-HQ face-attribute steering: 32+32 matched faces, 20 neutral prompts, seed 42",
        y=0.995,
    )
    fig.tight_layout()
    png = OUT / f"{TS}_celeba_alpha_curves.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")

    lines = [f"# CelebA face steering summary ({TS})", ""]
    lines.append("Runs: " + ", ".join(str(p.parent.parent) for p in RUNS.values()))
    lines.append(f"Alpha-curve plot: {png}")
    lines.append("")
    for attr, d in data.items():
        lines.append(
            f"## {attr} (face miss rate {d['face_miss_rate']:.1%}, probe bal_acc {d['probe']['val_metrics']['balanced_accuracy']:.3f})"
        )
        lines.append("")
        lines.append("| method | alpha | dP_probe | dP_clip | id_sim | faces found |")
        lines.append("|---|---|---|---|---|---|")
        cur = all_curves[attr]
        for method in ("DDIM", "MW"):
            for a in sorted(a for m, a in cur if m == method):
                rec = cur[(method, a)]
                idm, ids, idn = rec["d_id_sim"]
                id_str = f"{idm + 1.0:.3f}±{ids:.3f}" if not math.isnan(idm) else "nan"
                lines.append(
                    f"| {method} | {a:+.1f} | {fmt(rec['d_p_probe'])} | {fmt(rec['d_p_target'])}"
                    f" | {id_str} | {rec['found']:.0%} |"
                )
        lines.append("")
        lines.append("MW − DDIM paired on same (prompt, alpha):")
        for metric in ("p_probe", "p_target", "id_sim"):
            res = paired_mw_ddim(d["images"], metric)
            lines.append(f"- {metric}: pos-alpha {fmt(res['pos'])}, neg-alpha {fmt(res['neg'])}")
        lines.append("")
        drift = offtarget_drift(d["images"], attr)
        lines.append("Off-target probe drift at alpha=+2 (pooled MW+DDIM), delta vs base:")
        lines.append("| " + " | ".join(OFFTARGET) + " |")
        lines.append("|" + "---|" * len(OFFTARGET))
        lines.append("| " + " | ".join(fmt(drift[a]) for a in OFFTARGET) + " |")
        lines.append("")
    md = OUT / f"{TS}_celeba_summary.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
