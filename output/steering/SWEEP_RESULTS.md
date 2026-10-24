# 8-style sweep: image-derived steering boundary + DDIM vs MW (2026-10-24)

Runs `output/steering/sweep_<style>/20261024_*` at commit `130d35a`; scores in
each run's `scores_20261024_16*/`; aggregate tables in
`output/steering/sweep_summary/20261024_165226_sweep_summary.{md,json}`.

Setup per style: directions fit from 32 real WikiArt images vs 32 mixed
negatives (fresh download, seeds 101–108, min side 768), DDIM + MW inversion at
50 steps, empty prompts everywhere, uncond reads, SA injection into last 36
blocks with per-token renorm. Generation: 20 prompts × α ∈
{0.5,1,1.5,2,2.5,3} × {DDIM, MW directions} + base, seed 42, CFG 5 (260
images/style, 2080 total). Metrics: text-free DINOv2 probe P(style) (primary)
and CLIP zero-shot P(style) + CLIP-T, all paired vs the same-prompt base.

## Verdict per style (probe metric, best α)

| style | planned group | best dP_probe | α | P_probe base→best | CLIP P(target) best | works? |
|---|---|---|---|---|---|---|
| Fauvism | statistical | +0.151±0.023 | 2.5 (DDIM) | 0.028→0.178 | 0.41 | yes |
| Pointillism | statistical | +0.139±0.026 | 3.0 (MW) | 0.027→0.166 | 0.78 | yes |
| Impressionism | statistical | +0.127±0.021 | 2.5 (DDIM) | 0.072→0.199 | 0.42 | yes |
| Pop_Art | structural | +0.134±0.025 | 3.0 (DDIM) | 0.081→0.214 | 0.07 | yes |
| Cubism | structural | +0.118±0.014 | 2.5 (both) | 0.029→0.147 | 0.10 | yes |
| Ukiyo_e | structural | +0.048±0.017 | 1.5 (MW) | 0.026→0.075 | 0.04 | weak |
| Early_Renaissance | structural | +0.034±0.016 | 3.0 (MW) | 0.018→0.053 | 0.07 | weak |
| Color_Field_Painting | statistical | +0.015±0.006 | 2.5 (MW) | 0.023→0.039 | 0.20 | no |

CLIP-T cost ≤0.012 everywhere except cubism (−0.027 at α=2.5); prompt content
survives at every working operating point.

## Finding 1: the boundary is local-vs-global, not statistical-vs-structural

The planned statistical/structural split is falsified by its own test set:
Cubism and Pop Art ("structural" picks) steer as strongly as Fauvism, while
Color Field Painting (the easiest "statistical" pick on paper) fails. What the
winners share is identity carried by **local appearance statistics** —
brushwork, palette, stippling, faceted shading, flat saturated fills — which
channel-mean SA directions can encode. What the losers share is identity that
requires a **global change of composition or content**: full abstraction
(color field), period iconography (early renaissance), flat line-art
construction (ukiyo-e). Steering adds the look locally but cannot delete the
prompted object or re-stage the scene, and the probe correctly refuses to call
"a dog with big flat color patches" a color field painting.

Metric blind spots, both directions: CLIP overrates color field (0.20 vs probe
0.039 — it matches "color field" words, not the genre) and underrates pop art
(0.07 vs probe 0.21) and cubism (0.10 vs 0.15). Probe (per-class val acc
0.50–0.90 on these 8) is the metric of record.

## Finding 2: with 32 source images, MW ≈ DDIM

Paired MW−DDIM dP_probe at matched α, 20 prompts: only pointillism α=3 reaches
nominal significance (+0.021±0.010, t=2.16, 12/20); impressionism trends
DDIM-better (−0.016±0.004 at α=2, t=−1.83 at 2.5); the other six styles are
null (|t|<1.4). Earlier result at **16** source images (fauvism): MW beat DDIM
with paired t=3.68 (pass 2) and dP_probe +0.234 vs +0.192 (fine grid).

Interpretation (untested): averaging activations over 2×32 trajectories washes
out per-trajectory inversion-error differences, so inversion quality stops
mattering for the direction estimate; MW's advantage would then live in the
low-data regime. Confounds vs the earlier result: different source sets
(16-image seed-42/43 download vs 32-image seeds-101–108), 5 vs 20 prompts.

## Suggested next steps

1. Source-count ablation: refit fauvism/pointillism directions from n ∈ {4, 8,
   16, 32} subsets of the same sweep data, regenerate at the best α — directly
   tests whether MW>DDIM is a sample-efficiency effect.
2. Controls (swap/gaussian) for the 4 newly-working styles under the probe
   metric (cheap: reuse fits, 2 extra variants).
3. The local-vs-global boundary predicts structure-level steering needs a
   different injection site: attention maps / spatial patterns instead of
   channel means.
