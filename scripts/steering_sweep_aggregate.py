# ABOUTME: Aggregates the 8-style steering sweep score JSONs into one DDIM-vs-MW
# ABOUTME: comparison: per-style alpha curves, paired MW-DDIM deltas, group summary.
"""Aggregate sweep scores into SWEEP_RESULTS.md + JSON.

Usage:
    uv run python scripts/steering_sweep_aggregate.py \
        --sweep_root output/steering --out_dir output/steering/sweep_summary
"""

import glob
import json
import statistics
import subprocess
import time
from pathlib import Path

import fire

STYLE_GROUPS = {
    "statistical": ["fauvism", "pointillism", "impressionism", "color_field_painting"],
    "structural": ["ukiyo_e", "cubism", "early_renaissance", "pop_art"],
}
ALPHAS = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]
METRICS = ("csd", "p_probe", "p_target", "clip_t")
# First metric present in the scores, in METRICS order, picks best alphas and
# the headline paired MW-DDIM comparison.
PAIRED_METRICS = ("csd", "p_probe")


def sem(values: list[float]) -> float:
    return statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0


def latest_scores(sweep_root: Path, style: str) -> dict:
    pattern = str(sweep_root / f"sweep_{style}" / "2026*" / "scores_*" / "*_clip_scores.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return json.loads(Path(paths[-1]).read_text())


def paired_rows(images: list[dict]) -> dict[str, dict[str, dict[str, dict]]]:
    """Index image rows as [alpha][method][prompt] for paired comparisons."""
    index: dict[str, dict[str, dict[str, dict]]] = {}
    for row in images:
        if row["method"] == "base":
            continue
        alpha = str(row["strength"])
        index.setdefault(alpha, {}).setdefault(row["method"], {})[row["prompt"]] = row
    return index


def style_summary(scores: dict) -> dict:
    """Per-style alpha curves and paired MW-DDIM stats from one score JSON."""
    images = scores["images"]
    base = {row["prompt"]: row for row in images if row["method"] == "base"}
    index = paired_rows(images)
    metrics = [m for m in METRICS if m in images[0]]
    primary = metrics[0]
    curves: dict[str, dict[str, dict]] = {m: {} for m in ("DDIM", "MW")}
    paired: dict[str, dict] = {}
    for alpha in ALPHAS:
        for method in ("DDIM", "MW"):
            rows = index[alpha][method]
            assert set(rows) == set(base), (alpha, method)
            entry = {"n": len(rows)}
            for metric in metrics:
                deltas = [rows[p][metric] - base[p][metric] for p in base]
                entry[f"d_{metric}_mean"] = statistics.mean(deltas)
                entry[f"d_{metric}_sem"] = sem(deltas)
            curves[method][alpha] = entry
        paired[alpha] = {}
        for metric in [m for m in PAIRED_METRICS if m in metrics]:
            diffs = [index[alpha]["MW"][p][metric] - index[alpha]["DDIM"][p][metric] for p in base]
            mean, err = statistics.mean(diffs), sem(diffs)
            paired[alpha][metric] = {
                "mw_minus_ddim_mean": mean,
                "mw_minus_ddim_sem": err,
                "t": mean / err if err > 0 else 0.0,
                "wins": sum(d > 0 for d in diffs),
                "n": len(diffs),
            }
    best = {
        method: max(ALPHAS, key=lambda a: curves[method][a][f"d_{primary}_mean"])
        for method in ("DDIM", "MW")
    }
    return {"curves": curves, "paired": paired, "best_alpha": best, "primary": primary}


def fmt(mean: float, err: float) -> str:
    return f"{mean:+.3f}±{err:.3f}"


def to_markdown(summary: dict, meta: dict) -> str:
    primary = next(iter(summary.values()))["primary"]
    label = {"csd": "dCSD", "p_probe": "dP_probe"}[primary]
    lines = [
        "# 8-style sweep: DDIM vs MW steering (20 prompts, alpha 0.5-3.0)",
        "",
        f"Scored {meta['timestamp']} at commit `{meta['git_sha'][:8]}`. Primary "
        f"metric: {label} = paired change vs the unsteered base image (n=20 "
        "prompts, seed 42). CSD = mean style similarity to the 32 positive "
        "reference paintings; ceiling = mean ref-ref similarity, floor = mean "
        "negative-ref similarity. dCLIP-T = change in prompt adherence.",
        "",
        f"## Per-style best operating points (by {label})",
        "",
        f"| group | style | CSD floor/ceiling | base CSD | best α DDIM | {label} DDIM"
        f" | best α MW | {label} MW | MW−DDIM @ best MW α | t | wins |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, styles in STYLE_GROUPS.items():
        for style in styles:
            s = summary[style]
            bd, bm = s["best_alpha"]["DDIM"], s["best_alpha"]["MW"]
            cd, cm = s["curves"]["DDIM"][bd], s["curves"]["MW"][bm]
            p = s["paired"][bm][primary]
            bounds = (
                f"{s['csd_floor']:.2f}/{s['csd_ceiling']:.2f}"
                if s.get("csd_ceiling") is not None
                else "-"
            )
            base_csd = f"{s['base_csd']:.3f}" if s.get("base_csd") is not None else "-"
            lines.append(
                f"| {group} | {style} | {bounds} | {base_csd}"
                f" | {bd} | {fmt(cd[f'd_{primary}_mean'], cd[f'd_{primary}_sem'])}"
                f" | {bm} | {fmt(cm[f'd_{primary}_mean'], cm[f'd_{primary}_sem'])}"
                f" | {fmt(p['mw_minus_ddim_mean'], p['mw_minus_ddim_sem'])}"
                f" | {p['t']:.2f} | {p['wins']}/{p['n']} |"
            )
    header = "| style | " + " | ".join(f"α={a}" for a in ALPHAS) + " |"
    for metric in next(iter(summary.values()))["paired"][ALPHAS[0]]:
        metric_label = {"csd": "dCSD", "p_probe": "dP_probe"}[metric]
        lines += ["", f"## Paired MW−DDIM {metric_label} per alpha", ""]
        lines += [header, "|---|" + "---|" * len(ALPHAS)]
        for group, styles in STYLE_GROUPS.items():
            for style in styles:
                cells = []
                for alpha in ALPHAS:
                    p = summary[style]["paired"][alpha][metric]
                    cells.append(fmt(p["mw_minus_ddim_mean"], p["mw_minus_ddim_sem"]))
                lines.append(f"| {style} | " + " | ".join(cells) + " |")
    present = [
        (m, lbl)
        for m, lbl in (("csd", "dCSD"), ("p_probe", "dP_probe"), ("clip_t", "dCLIP-T"))
        if f"d_{m}_mean" in next(iter(summary.values()))["curves"]["DDIM"][ALPHAS[0]]
    ]
    for metric, metric_label in present:
        lines += [
            "",
            f"## {metric_label} per alpha (DDIM / MW)",
            "",
            header,
            "|---|" + "---|" * len(ALPHAS),
        ]
        for group, styles in STYLE_GROUPS.items():
            for style in styles:
                cells = []
                for alpha in ALPHAS:
                    d = summary[style]["curves"]["DDIM"][alpha][f"d_{metric}_mean"]
                    m = summary[style]["curves"]["MW"][alpha][f"d_{metric}_mean"]
                    cells.append(f"{d:+.3f} / {m:+.3f}")
                lines.append(f"| {style} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(
    sweep_root: str = "output/steering",
    out_dir: str = "output/steering/sweep_summary",
    probe_path: str = "output/steering/probe/wikiart_dinov2_probe.pt",
) -> None:
    """Aggregate all sweep_<style> score JSONs and write markdown + JSON summary."""
    import torch

    root = Path(sweep_root)
    per_class = torch.load(probe_path, weights_only=False)["val_accuracy_per_class"]
    summary: dict[str, dict] = {}
    for styles in STYLE_GROUPS.values():
        for style in styles:
            scores = latest_scores(root, style)
            s = style_summary(scores)
            s["target_style"] = scores["target_style"]
            s["run_dir"] = scores["run_dir"]
            s["scores_timestamp"] = scores["timestamp"]
            s["probe_class_acc"] = per_class[scores["target_style"]]
            csd_meta = scores.get("csd")
            s["csd_ceiling"] = csd_meta["ref_self_sim"] if csd_meta else None
            s["csd_floor"] = csd_meta["neg_ref_sim"] if csd_meta else None
            base_rows = [r for r in scores["images"] if r["method"] == "base"]
            s["base_csd"] = (
                statistics.mean(r["csd"] for r in base_rows) if "csd" in base_rows[0] else None
            )
            summary[style] = s
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {"timestamp": timestamp, "git_sha": commit, "probe_path": probe_path}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{timestamp}_sweep_summary.json").write_text(
        json.dumps({"meta": meta, "styles": summary}, indent=2) + "\n"
    )
    markdown = to_markdown(summary, meta)
    (out / f"{timestamp}_sweep_summary.md").write_text(markdown)
    print(markdown)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
