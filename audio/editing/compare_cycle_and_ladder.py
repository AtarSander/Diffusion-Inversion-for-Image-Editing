# ABOUTME: Reads the two scored Stable Audio sweeps -- the cycle-loss arms at step 2000 and the
# ABOUTME: editing half of the accuracy ladder -- into tables, so both questions get an answer.

import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import pandas as pd

AUDIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_ROOT))
sys.path.insert(0, str(AUDIO_ROOT / "editing"))
from run_metrics import PER_EXAMPLE_CSV  # noqa: E402

METRICS = {"lpaps": "lpaps", "clap": "clap", "muq": "muqt_sim_p0", "clap_dir": "clap_dir"}
CYCLE = re.compile(
    r"stableaudio_odeinvlora_hparam_saocos_(?P<arm>.+?)_checkpoint_step_(?P<step>\d+)"
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

    out = AUDIO_ROOT / out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
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
