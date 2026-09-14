# ABOUTME: Reads the two scored Stable Audio sweeps -- the cycle-loss arms at step 2000 and the
# ABOUTME: editing half of the accuracy ladder -- into tables, so both questions get an answer.

import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))
sys.path.insert(0, str(AUDIO_ROOT / "editing"))
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir"}
NOLORA = re.compile(
    r"stableaudio_odeinv_nolora_hparam_cfgtar(?P<cfg>[\d.]+)_t(?P<tstart>\d+)_s\d+$"
)
# step_2000 is pinned, not captured loosely: the earlier hparam sweep also wrote
# ..._checkpoint_step_4000_... runs for the same training run, and a pattern that accepted any
# step pooled them under one label, which pivot_table then averaged into a baseline ~0.15 LPAPS
# worse than it is -- manufacturing exactly the improvement the multi-step arms appeared to have.
CYCLE = re.compile(
    r"stableaudio_odeinvlora_hparam_saocos_(?P<arm>.+?)_checkpoint_step_(?P<step>2000)"
    r"_cfgtar(?P<cfg>[\d.]+)_t(?P<tstart>\d+)_s\d+$"
)
LADDER = re.compile(
    r"stableaudio_acc_edit_hparam_cfgtar(?P<cfg>[\d.]+)_t(?P<tstart>\d+)_s\d+"
    r"_(?:nolora|saocos_r8_a4_lr5e-5_checkpoint_step_(?P<step>\d+)(?P<ema>_ema)?)$"
)


def rows_for(root: Path, pattern: re.Pattern, label) -> pd.DataFrame:
    """Collect scored runs matching a pattern into one row each."""
    out = []
    for run in sorted(root.glob("stableaudio_*")):
        m = pattern.match(run.name)
        csv = run / PER_EXAMPLE_CSV
        if not m or not csv.exists():
            continue
        frame = pd.read_csv(csv)
        row = {"label": label(m), "tstart": int(m.group("tstart")),
               "cfg": float(m.group("cfg")), "n": len(frame)}
        for name, col in METRICS.items():
            row[name] = frame[col].mean()
        out.append(row)
    return pd.DataFrame(out)


def main(runs_root: str, out_root: str = "output/cycle_and_ladder") -> None:
    """Print and save both comparisons.

    Args:
        runs_root: Directory holding the stable_audio run directories.
        out_root: Destination, relative to `audio/`.
    """
    root = Path(runs_root)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    cycle = rows_for(root, CYCLE, lambda m: {"r8_a4_lr5e-5": "baseline (no cycle)"}.get(
        m.group("arm"), m.group("arm").replace("cyc_", "").replace("_g1.0_r8_a4_lr5e-5", "")))
    cycle = cycle.sort_values(["label", "tstart"])
    cycle["arm_label"] = cycle["label"]
    dupes = cycle.duplicated(["label", "tstart", "cfg"], keep=False)
    assert not dupes.any(), (
        "two runs share an (arm, tstart, cfg) cell, so the table would average them:\n"
        f"{cycle[dupes][['label', 'tstart', 'cfg', 'lpaps']]}"
    )
    print("=== CYCLE ARMS, all at step 2000, cfg_tar 3.5 ===")
    print(cycle.to_string(index=False, float_format=fmt))
    pivot = cycle.pivot_table(index="tstart", columns="label", values="lpaps")
    if "baseline (no cycle)" in pivot:
        print("\nLPAPS minus baseline (negative = better preserved):")
        print((pivot.sub(pivot["baseline (no cycle)"], axis=0)
               .drop(columns="baseline (no cycle)")).to_string(float_format=lambda v: f"{v:+.4f}"))

    def ladder_label(m):
        if m.group("step") is None:
            return "no LoRA"
        return f"step {int(m.group('step')):>5}{' EMA' if m.group('ema') else ''}"

    ladder = rows_for(root, LADDER, ladder_label).sort_values(["tstart", "label"])
    print("\n=== ACCURACY LADDER, editing half (cfg_tar 3.5) ===")
    for t, sub in ladder.groupby("tstart"):
        print(f"\n-- tstart {t} --")
        print(sub[["label", "lpaps", "clap", "muq", "clap_dir"]]
              .to_string(index=False, float_format=fmt))

    # The cycle grid has no no-LoRA arm of its own; the earlier sweep ran it at identical
    # settings, so it supplies the reference the improvements are measured against.
    # Restricted to the guidance the cycle arms ran at: the earlier sweep also covers cfg 7.0,
    # and keeping both puts two rows on every tstart, so the lookup returns a Series.
    nolora = rows_for(root, NOLORA, lambda m: "no LoRA")
    cycle_cfg = float(cycle["cfg"].iloc[0])
    nolora = nolora[nolora.cfg == cycle_cfg].drop_duplicates("tstart").set_index("tstart")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    fig.suptitle(
        "Stable Audio: editing saturates by step 2000, and the multi-step cycle loss is the "
        "thing that moves it", fontsize=13.5, fontweight="bold", y=0.99,
    )

    # (a), (b): the ladder is flat in both metrics.
    for ax, metric, name, better in (
        (axes[0, 0], "lpaps", "LPAPS to source", "lower better"),
        (axes[0, 1], "clap", "CLAP to target caption", "higher better"),
    ):
        for t, color in ((50, "#1f77b4"), (75, "#d62728")):
            sub = ladder[(ladder.tstart == t) & ladder.label.str.startswith("step")].copy()
            sub["step"] = sub.label.str.extract(r"(\d+)").astype(int)
            sub = sub[~sub.label.str.contains("EMA")].sort_values("step")
            ax.plot(sub["step"], sub[metric], "o-", color=color, ms=5, lw=1.8,
                    label=f"tstart {t}, adapter")
            ref = ladder[(ladder.tstart == t) & (ladder.label == "no LoRA")][metric]
            if len(ref):
                ax.axhline(float(ref.iloc[0]), color=color, ls="--", lw=1.4, alpha=0.7,
                           label=f"tstart {t}, no LoRA")
        ax.set(xlabel="training step", ylabel=name, xscale="log",
               title=f"({'ab'[metric != 'lpaps']}) Ladder: {name} ({better})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8.5)

    # (c): how much each arm beats no LoRA, so the baseline's own gain sets the scale.
    ax = axes[1, 0]
    tstarts = sorted(cycle.tstart.unique())
    arms = ["baseline (no cycle)", "k2", "k3"]
    width, colors = 0.26, {"baseline (no cycle)": "#7f7f7f", "k2": "#d62728", "k3": "#ff7f0e"}
    for i, arm in enumerate(arms):
        gains = []
        for t in tstarts:
            row = cycle[(cycle.arm_label == arm) & (cycle.tstart == t)]
            base = nolora.loc[t, "lpaps"] if t in nolora.index else float("nan")
            gains.append(base - float(row["lpaps"].iloc[0]) if len(row) else float("nan"))
        ax.bar([x + (i - 1) * width for x in range(len(tstarts))], gains, width,
               color=colors[arm], label=arm)
    ax.set(xlabel="tstart", ylabel="LPAPS improvement over no LoRA",
           title="(c) Cycle arms at step 2000: multi-step roughly doubles the gain")
    ax.set_xticks(range(len(tstarts)))
    ax.set_xticklabels(tstarts)
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    # (d): the front, so a preservation gain cannot be hiding an alignment loss.
    ax = axes[1, 1]
    for arm in arms:
        sub = cycle[cycle.arm_label == arm].sort_values("lpaps")
        ax.plot(sub["lpaps"], sub["clap"], "o-", color=colors[arm], ms=6, lw=1.7, label=arm)
        for _, r in sub.iterrows():
            ax.annotate(f"t{int(r['tstart'])}", (r["lpaps"], r["clap"]), fontsize=7.5,
                        textcoords="offset points", xytext=(4, 4), color=colors[arm])
    if len(nolora):
        ref = nolora.reset_index().sort_values("lpaps")
        ax.plot(ref["lpaps"], ref["clap"], "s--", color="k", ms=5, lw=1.4, alpha=0.6,
                label="no LoRA")
    ax.set(xlabel="LPAPS to source (lower = better preserved)", ylabel="CLAP to target caption",
           title="(d) Cycle arms: the front, not just one metric")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"cycle_and_ladder.{ext}", dpi=150, bbox_inches="tight")
    cycle.to_csv(out / "cycle_arms.csv", index=False)
    ladder.to_csv(out / "accuracy_ladder_edit.csv", index=False)
    lines = ["# Cycle arms and the editing half of the accuracy ladder\n",
             "## Cycle arms (step 2000, cfg_tar 3.5)\n",
             cycle.to_markdown(index=False, floatfmt=".4f"),
             "\n## Accuracy ladder, editing (cfg_tar 3.5)\n",
             ladder.to_markdown(index=False, floatfmt=".4f"), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
