# ABOUTME: Emit the LaTeX table for the real-audio DDIM reconstruction comparison, values with
# ABOUTME: SEM plus a companion table of the paired differences against the no-LoRA baseline.

from pathlib import Path

import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "audio"))

from editing.run_metrics import per_example  # noqa: E402

ROOT = Path(
    "/nas/lstanisz/code/lorainv/pwr-mount/audio/outputs/edits/medleymd/audioldm2_ddim"
)
STEPS = ["200", "50"]
VARIANTS = [("nolora", "DDIM"), ("attn", "DDIM + LoRA (attn)"), ("full", "DDIM + LoRA (full)")]
METRICS = [("LPAPS", "lpaps", "lower"), ("psnr", "psnr", "higher"), ("ssim", "ssim", "higher")]


def load(step: str, variant: str, column: str) -> pd.Series:
    """One metric for one reconstruction run, keyed so the variants can be paired."""
    return per_example(ROOT / f"recon_tracks_s{step}_{variant}", column)


data = {
    (step, variant): {m: load(step, variant, c) for m, c, _ in METRICS}
    for step in STEPS
    for variant, _ in VARIANTS
}

print(r"""% requires: \usepackage{booktabs,multirow}
\begin{table}[t]
  \centering
  \caption{DDIM reconstruction of real audio: 35 distinct MedleyDB tracks inverted and denoised
    with the same source caption at $w_{\mathrm{src}} = w_{\mathrm{tar}} = 1$, so the output
    should be the input. Scored against the input through the same pipeline as the editing
    benchmark, so the absolute values include vocoder and bandwidth loss that inversion does not
    control. Subscripts are standard errors over the 35 tracks; best per panel in bold.}
  \label{tab:medleydb-reconstruction}
  \begin{tabular}{ll ccc}
    \toprule
    Steps & Method & LPAPS $\downarrow$ & mel PSNR $\uparrow$ & mel SSIM $\uparrow$ \\
    \midrule""")

for step in STEPS:
    best = {}
    for metric, _, direction in METRICS:
        pick = min if direction == "lower" else max
        best[metric] = pick(VARIANTS, key=lambda v: data[(step, v[0])][metric].mean())[0]
    print(f"    \\multirow{{3}}{{*}}{{{step}}}")
    for variant, label in VARIANTS:
        cells = []
        for metric, _, _ in METRICS:
            series = data[(step, variant)][metric]
            digits = 3 if metric != "psnr" else 3
            body = f"{series.mean():.{digits}f}"
            if variant == best[metric]:
                body = f"\\mathbf{{{body}}}"
            cells.append(f"${body}_{{\\pm {series.sem():.3f}}}$")
        print(f"      & {label:20s} & " + " & ".join(cells) + r" \\")
    print(r"    \midrule" if step != STEPS[-1] else r"    \bottomrule")

print(r"""  \end{tabular}
\end{table}""")

print(r"""
\begin{table}[t]
  \centering
  \caption{Paired difference against the no-LoRA reconstruction, over the same 35 tracks.
    Pairing is what makes an effect this small resolvable: the spread across tracks is an order
    of magnitude larger than every difference here. Brackets are 95\% confidence intervals.}
  \label{tab:medleydb-reconstruction-paired}
  \begin{tabular}{ll ccc}
    \toprule
    Steps & Adapter & $\Delta$LPAPS & $\Delta$mel PSNR & $\Delta$mel SSIM \\
    \midrule""")

for step in STEPS:
    print(f"    \\multirow{{2}}{{*}}{{{step}}}")
    for variant, label in VARIANTS[1:]:
        cells = []
        for metric, _, _ in METRICS:
            base = data[(step, "nolora")][metric]
            other = data[(step, variant)][metric]
            common = base.index.intersection(other.index)
            delta = other.loc[common] - base.loc[common]
            _, p = stats.ttest_rel(other.loc[common], base.loc[common])
            ci = 1.96 * delta.std() / len(delta) ** 0.5
            star = "^{*}" if p < 0.05 else ""
            cells.append(
                f"${delta.mean():+.4f}{star}$ {{\\scriptsize $[{delta.mean()-ci:+.4f}, "
                f"{delta.mean()+ci:+.4f}]$}}"
            )
        name = label.split("(")[1].rstrip(")")
        print(f"      & {name:6s} & " + " & ".join(cells) + r" \\")
    print(r"    \midrule" if step != STEPS[-1] else r"    \bottomrule")

print(r"""  \end{tabular}

  \vspace{0.4em}
  \parbox{\linewidth}{\footnotesize $^{*}$ $p < 0.05$, paired $t$-test over the 35 tracks.}
\end{table}""")
