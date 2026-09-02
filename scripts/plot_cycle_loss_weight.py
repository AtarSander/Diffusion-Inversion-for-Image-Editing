# ABOUTME: Plots the measured cycle-loss weight diagnostics from measure_cycle_loss_weight.py:
# ABOUTME: ratio vs timestep, loss magnitudes, residual equivalence, contraction, grid-length sweep.

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C_INV, C_CYC, C_REF = "#1f77b4", "#d62728", "#7f7f7f"


def load(results_dir: Path) -> dict[int, dict]:
    """Load every measurement JSON in the directory, keyed by grid length."""
    out = {}
    for p in sorted(results_dir.glob("cycle_weight_sd14_ddim*.json")):
        d = json.load(open(p))
        out[d["num_ddim_steps"]] = d  # later timestamp for the same N wins
    assert out, f"no measurement JSONs in {results_dir}"
    return out


def by_timestep(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a per-transition field across prompts, returning (t, mean, spread)."""
    acc = defaultdict(list)
    for r in rows:
        acc[r["t"]].append(r[key])
    t = np.array(sorted(acc))
    vals = [np.array(acc[int(x)]) for x in t]
    return t, np.array([v.mean() for v in vals]), np.array([v.std() for v in vals])


def main(
    results_dir: str = "output/cycle_loss_weight",
    main_steps: int = 50,
    output_dir: str = "output/cycle_loss_weight/plots",
) -> None:
    """Draw the five-panel diagnostic figure.

    Args:
        results_dir: Where the measurement JSONs live, relative to the repo root.
        main_steps: Grid length used for the per-timestep panels.
        output_dir: Destination for the PNG/SVG, relative to the repo root.
    """
    data = load(ROOT / results_dir)
    assert main_steps in data, f"no run at {main_steps} steps; have {sorted(data)}"
    d = data[main_steps]
    rows, s = d["rows"], d["summary"]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    fig.suptitle(
        f"Cycle loss vs $\\mathcal{{L}}_{{inv}}$ — measured on {d['model_key'].split('/')[-1]}, "
        f"DDIM-{main_steps}, cfg=1, $\\lambda_{{cycle}}$=1.0",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.935,
        f"mean ratio {100 * s['mean_ratio']:.3f}%  (1 part in {1 / s['mean_ratio']:.0f})   |   "
        f"$\\lambda_{{cycle}}$ for parity ≈ {1 / s['mean_ratio']:.0f}   |   "
        f"{len(rows)} transitions, {len(set(r['prompt'] for r in rows))} prompts, fp32",
        ha="center", fontsize=10.5, color="#333",
    )

    # (a) the headline: ratio vs timestep, with B^2 overlaid
    ax = axes[0, 0]
    t, m, sd = by_timestep(rows, "ratio")
    tb, b2, _ = by_timestep(rows, "B2")
    ax.fill_between(t, m - sd, m + sd, color=C_CYC, alpha=0.20, lw=0)
    ax.plot(t, m, "o-", color=C_CYC, ms=3.5, lw=1.6, label=r"measured $\mathcal{L}_{cyc}/\mathcal{L}_{inv}$")
    ax.plot(tb, b2, "--", color="k", lw=1.4, label=r"$B^2$ (analytic prediction)")
    ax.axhline(s["mean_ratio"], color=C_REF, ls=":", lw=1.3,
               label=f"mean = {100 * s['mean_ratio']:.3f}%")
    ax.set(yscale="log", xlabel="timestep $t_i$", ylabel="ratio",
           title="(a) Effective weight of the cycle term")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) absolute magnitudes
    ax = axes[0, 1]
    for key, c, lab in (("L_inv", C_INV, r"$\mathcal{L}_{inv}$"),
                        ("L_cycle", C_CYC, r"$\lambda\mathcal{L}_{cycle}$, $\lambda$=1")):
        t, m, sd = by_timestep(rows, key)
        ax.fill_between(t, np.maximum(m - sd, 1e-12), m + sd, color=c, alpha=0.20, lw=0)
        ax.plot(t, m, "o-", color=c, ms=3.5, lw=1.6, label=lab)
    ax.set(yscale="log", xlabel="timestep $t_i$", ylabel="loss value (MSE)",
           title="(b) Both terms, absolute scale")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) residual equivalence: why the ratio IS B^2
    ax = axes[0, 2]
    t, m, sd = by_timestep(rows, "resid_ratio")
    ax.fill_between(t, m - sd, m + sd, color="#2ca02c", alpha=0.20, lw=0)
    ax.plot(t, m, "o-", color="#2ca02c", ms=3.5, lw=1.6)
    ax.axhline(1.0, color="k", ls="--", lw=1.3, label="1.0 (identical residuals)")
    ax.axhline(s["mean_resid_ratio"], color=C_REF, ls=":", lw=1.3,
               label=f"mean = {s['mean_resid_ratio']:.4f}")
    ax.set(xlabel="timestep $t_i$", ylabel=r"$\|e-v\| \, / \, \|e-u\|$",
           title="(c) The two terms share one residual")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # (d) contraction factor -> uniqueness of the zero
    ax = axes[1, 0]
    contr = [abs(r["B"] / r["A"]) * r["J_eff"] for r in rows]
    for r, c in zip(rows, contr):
        r["_contr"] = c
    t, m, sd = by_timestep(rows, "_contr")
    ax.fill_between(t, m - sd, m + sd, color="#9467bd", alpha=0.20, lw=0)
    ax.plot(t, m, "o-", color="#9467bd", ms=3.5, lw=1.6, label=r"$|B/A|\cdot\|J\|$")
    ax.axhline(1.0, color=C_CYC, ls="--", lw=1.6, label="1.0 — contraction fails above")
    ax.set(yscale="log", xlabel="timestep $t_i$", ylabel="contraction factor",
           title=f"(d) Zero is unique (mean {np.mean(contr):.3f} $\\ll$ 1)")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # (e) effective Jacobian norm
    ax = axes[1, 1]
    t, m, sd = by_timestep(rows, "J_eff")
    ax.fill_between(t, m - sd, m + sd, color="#ff7f0e", alpha=0.20, lw=0)
    ax.plot(t, m, "o-", color="#ff7f0e", ms=3.5, lw=1.6)
    ax.axhline(1 / max(abs(r["B"] / r["A"]) for r in rows), color=C_CYC, ls="--", lw=1.4,
               label=r"$\|J\|$ needed to break contraction")
    ax.set(xlabel="timestep $t_i$", ylabel=r"$\|J\|_{eff}=\|v-u\|/\|\hat z_i-z_i\|$",
           title="(e) Teacher Jacobian is well-behaved")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # (f) grid-length sweep: when does the term stop being negligible?
    ax = axes[1, 2]
    ns = sorted(data)
    ratios = [data[n]["summary"]["mean_ratio"] for n in ns]
    ax.plot(ns, [100 * r for r in ratios], "o-", color=C_CYC, ms=7, lw=1.8)
    for n, r in zip(ns, ratios):
        ax.annotate(f"{100 * r:.2f}%\n$\\lambda$≈{1 / r:.0f}", (n, 100 * r),
                    textcoords="offset points", xytext=(6, 6), fontsize=8.5)
    ax.axhline(100, color="k", ls="--", lw=1.3, label="parity with $\\mathcal{L}_{inv}$")
    ax.set(xscale="log", yscale="log", xlabel="DDIM grid length (steps)",
           ylabel="mean ratio (%)", title="(f) Weight grows as the grid coarsens")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.925])
    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ext in ("png", "svg"):
        p = out / f"cycle_loss_weight_sd14_{stamp}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    fire.Fire(main)
