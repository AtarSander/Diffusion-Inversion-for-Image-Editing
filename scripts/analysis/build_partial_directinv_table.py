"""Build the SD1.5 partial-depth comparison from completed metric summaries."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = [
    ("structure_distance", 1000, False),
    ("psnr_unedit_part", 1, True),
    ("lpips_unedit_part", 1000, False),
    ("mse_unedit_part", 10000, False),
    ("ssim_unedit_part", 100, True),
    ("clip_similarity_target_image", 1, True),
    ("clip_similarity_target_image_edit_part", 1, True),
]
BASE = Path("checkpoints/../evaluation/sd15_fp16_baselines/metrics")
FULL = Path("checkpoints/../evaluation/one_epoch/cycle_loss_cfg1_r8_uniform_e1_all_editors/metrics")
IMPORTED = Path("reports/sd15_partial_normal_loss.tex")


def imported_rows(editor):
    text = (ROOT / IMPORTED).read_text()
    block = text.split(r"\label{tab:near-full-depth-" + editor + "}", 1)[1].split(r"\end{table}", 1)[0]
    rows = []
    for line in block.splitlines():
        if " & " not in line or r"\%" not in line:
            continue
        cells = line.split(" & ")
        label = re.sub(r"\\rowcolor\{[^}]+\}", "", cells[0]).strip()
        percent = int(re.search(r"\d+", cells[1]).group())
        values = [float(re.search(r"\d+\.\d+", cell).group()) / scale
                  for cell, (_, scale, _) in zip(cells[2:], METRICS)]
        assert len(values) == 7
        rows.append((label, percent, values))
    return rows


def read(path, method):
    if not (ROOT / path).is_file():
        return None
    summary = json.loads((ROOT / path).read_text())
    if summary["num_images"] != 700:
        raise ValueError(f"Incomplete evaluation: {path}")
    values = [summary["metrics"][f"{method}|{key}"]["mean"] for key, _, _ in METRICS]
    if not all(v is not None and math.isfinite(v) for v in values):
        raise ValueError(f"Invalid metrics: {path}")
    return values


def main():
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[a4paper,landscape,margin=1.2cm]{geometry}",
             r"\usepackage{booktabs,graphicx}",
             r"\usepackage[table]{xcolor}",
             r"\definecolor{allbetter}{RGB}{220,242,220}", r"\begin{document}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\renewcommand{\arraystretch}{1.10}"]
    provenance = []
    for editor, title in [("p2p", "P2P"), ("pnp", "PnP"), ("masactrl", "MasaCtrl"), ("pix2pix-zero", "Pix2Pix-Zero")]:
        lines.extend([
            r"\begin{table}[!htbp]\centering\footnotesize",
            r"\caption{" + title + r" on PIE-Bench with SD1.5. Depth is the retained fraction of 50 inversion steps. Moonwalker uses standard loss (r16) or cycle loss (r8, $\lambda=1$). Bold marks the best displayed value. Underlining marks improvement over 100\% DirectInv; green rows improve all seven metrics.}",
            r"\label{tab:sd15-partial-directinv-" + editor + "}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrr}\toprule",
            r"Inversion & Depth & Structure $\downarrow$ & PSNR $\uparrow$ & LPIPS $\downarrow$ & MSE $\downarrow$ & SSIM $\uparrow$ & CLIP target $\uparrow$ & CLIP edit $\uparrow$\\",
            r" & & $\times10^3$ & & $\times10^3$ & $\times10^4$ & $\times10^2$ & &\\\midrule",
        ])
        rows = []
        def add(label, percent, path, method):
            values = read(path, method)
            if values is not None:
                rows.append((label, percent, values))
            provenance.append(dict(editor=editor, method=label, percent=percent,
                                   source=str(path), available=values is not None))
        add("DDIM", 100, BASE / f"ddim+{editor}_summary.json", f"ddim+{editor}")
        add("DirectInv (baseline)", 100, BASE / f"directinversion+{editor}_summary.json", f"directinversion+{editor}")
        for percent in range(90, 101, 2):
            if percent < 100:
                add("DirectInv", percent, Path(f"data/evaluation/sd15_directinv_partial/steps{percent//2}/metrics/directinversion+{editor}_summary.json"), f"directinversion+{editor}")
            # DirectInv at 100% is the baseline row, not a duplicate experiment.
            path = FULL / f"lora+directinversion+{editor}_summary.json" if percent == 100 else Path(f"data/evaluation/sd15_cycle_r8_lambda1_partial/steps{percent//2}/metrics/lora+directinversion+{editor}_summary.json")
            add("Moonwalker (cycle loss, r8) + DirectInv", percent, path, f"lora+directinversion+{editor}")
        baseline = rows[1][2]
        extra = [row for row in imported_rows(editor) if row[0].startswith("Moonwalker")]
        rows.extend(extra)
        # One baseline pair, followed by depth-matched methods in a common table.
        rows = rows[:2] + sorted(rows[2:], key=lambda row: (
            row[1],
            0 if row[0] == "DirectInv" else
            1 if " + DirectInv" not in row[0] else
            2 if "standard loss" in row[0] else 3,
        ))
        for label, percent, values in extra:
            provenance.append(dict(editor=editor, method=label, percent=percent,
                                   source=str(IMPORTED), available=True, precision="two decimals"))
        best = [(max if higher else min)(round(r[2][i] * METRICS[i][1], 2) for r in rows)
                for i, (_, _, higher) in enumerate(METRICS)]
        for index, (label, percent, values) in enumerate(rows):
            if index > 2 and percent != rows[index - 1][1]:
                lines.append(r"\addlinespace[3pt]")
            cells = []
            for i, (_, scale, higher) in enumerate(METRICS):
                if values is None:
                    cells.append("--")
                    continue
                value = values[i]
                cell = f"{value * scale:.2f}"
                if round(value * scale, 2) == best[i]:
                    cell = r"\mathbf{" + cell + "}"
                if baseline is not None and (value > baseline[i] if higher else value < baseline[i]):
                    cell = r"\underline{" + cell + "}"
                cells.append("$" + cell + "$")
            all_better = baseline is not None and all(
                values[i] > baseline[i] if higher else values[i] < baseline[i]
                for i, (_, _, higher) in enumerate(METRICS)
            )
            prefix = r"\rowcolor{allbetter} " if all_better else ""
            lines.append(prefix + " & ".join([label, f"{percent}\\%", *cells]) + r"\\")
            if all_better:
                print(f"All metrics improved: {title}, {label}, {percent}%")
            if index == 1:
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule\end{tabular}%", "}", r"\end{table}", r"\clearpage"])
    lines.append(r"\end{document}")
    target = ROOT / "reports/sd15_partial_directinv_comparison.tex"
    target.write_text("\n".join(lines) + "\n")
    target.with_suffix(".sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {target}: {sum(p['available'] for p in provenance)}/{len(provenance)} rows available")


if __name__ == "__main__":
    main()
