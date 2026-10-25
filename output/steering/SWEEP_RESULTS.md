# 8-style sweep: image-derived steering boundary + DDIM vs MW (2026-10-24/25)

Runs `output/steering/sweep_<style>/20261024_*` at commit `130d35a`; CSD+probe+CLIP
scores (2026-10-25, commit `9e3aaa4`) in each run's `scores_20261025_*/`; aggregate
tables in `output/steering/sweep_summary/20261025_135431_sweep_summary.{md,json}`.

Setup per style: directions fit from 32 real WikiArt images vs 32 mixed
negatives (fresh download, seeds 101–108, min side 768), DDIM + MW inversion at
50 steps, empty prompts everywhere, uncond reads, SA injection into last 36
blocks with per-token renorm. Generation: 20 prompts × α ∈
{0.5,1,1.5,2,2.5,3} × {DDIM, MW directions} + base, seed 42, CFG 5 (260
images/style, 2080 total). Metrics, all paired vs the same-prompt base:
**CSD** (Somepalli et al. 2024, official ViT-L via `steering/csd.py`) = mean
style similarity to the 32 reference paintings — primary; DINOv2 probe
P(style) = 27-way class membership; CLIP zero-shot P(style); CLIP-T.

## Verdict per style (best α by dCSD)

| style | planned group | base→best CSD (ceiling) | dCSD | dP_probe @ its best α | verdict |
|---|---|---|---|---|---|
| Pointillism | statistical | 0.00→0.45 (0.46) | +0.446 | +0.139 | works |
| Impressionism | statistical | 0.10→0.31 (0.33) | +0.210 | +0.127 | works |
| Fauvism | statistical | 0.01→0.16 (0.35) | +0.154 | +0.151 | works |
| Pop_Art | structural | 0.03→0.18 (0.26) | +0.149 | +0.134 | works |
| Cubism | structural | 0.05→0.18 (0.32) | +0.126 | +0.118 | works |
| Ukiyo_e | structural | 0.06→0.22 (0.44) | +0.161 | +0.048 | style-only |
| Early_Renaissance | structural | 0.09→0.23 (0.48) | +0.142 | +0.034 | style-only |
| Color_Field_Painting | statistical | 0.02→0.17 (0.37) | +0.154 | +0.015 | style-only |

Ceiling = mean ref-ref CSD similarity; floors (neg-ref) are 0.11–0.19.
CLIP-T cost ≤0.012 everywhere except cubism (−0.027); content survives at
every operating point. Pointillism nearly saturates its ceiling (98% of
base→ceiling headroom); impressionism 89%; the rest land at 36–65%.

## Finding 1 (revised by CSD): steering always moves style; class membership is the bottleneck

Under the probe alone, 3 styles looked like failures. CSD shows **all 8 styles
move substantially toward the reference style manifold** (dCSD +0.13..+0.45,
i.e. 36–98% of the available headroom) — including Color Field,
Early Renaissance and Ukiyo-e. The correct statement of the boundary: SA
channel-mean steering reliably transfers reference **appearance statistics**
(CSD), but for styles whose class identity requires global composition or
content change (full abstraction, period iconography, line-art construction)
that is not sufficient for **class membership** (probe stays low) — a steered
"a dog" gains color-field-like fields yet is still a dog. Statistical styles
pass both tests; "structural" styles pass CSD only.

Metric blind spots mapped: CLIP zero-shot overrates color field (word match)
and underrates pop art/cubism ~2–3×; the probe penalizes styles whose class
boundary is content-defined. CSD-to-references is the fairest primary metric
and matches the paper's style-transfer protocol.

## Finding 2 (revised by CSD): MW beats DDIM exactly on the hard styles

Paired MW−DDIM dCSD at matched α: **Ukiyo-e +0.015±0.004 (t=3.69, 17/20) at
α=2**, Color Field +0.006±0.003 (t=2.01, 15/20), Cubism +0.007±0.004 (t=1.90,
16/20; +0.012 at α=1.5) — the three styles farthest from SDXL's prompted
prior. On the easy statistical styles the sign flips slightly (fauvism
−0.007, t=−2.82; impressionism −0.011..−0.014 at α≥2.5). Effects are small
(|Δ|≤0.015) next to the steering effect itself (+0.13..+0.45).

Revised interpretation: with 32 source images, inversion quality no longer
matters for styles SDXL renders easily, but **MW still extracts better
directions where inversion is hard** (flat line-art, abstraction — regimes
where DDIM inversion drifts). This refines, rather than contradicts, the
16-image result (MW t=3.68 on fauvism): more sources wash out the MW edge on
easy styles; the edge persists on hard ones. The probe agrees directionally on
ukiyo-e (MW best everywhere it works).

## Suggested next steps

1. Source-count ablation n ∈ {4,8,16,32} on fauvism + ukiyo_e (same sweep
   data): tests the sample-efficiency account of the MW edge.
2. Inversion-difficulty correlate: per-style DDIM reconstruction error vs
   MW−DDIM dCSD across the 8 styles (data already on disk from extract).
3. Controls (swap/gaussian) under CSD for 2–3 styles.
4. Structure-level injection (attention maps / spatial patterns) for the
   class-membership gap on style-only styles.
