# Image-derived steering, pass 2: controls + seeds (2026-10-22)

SA representation only, alpha {2, 4}, controls {true, swap, gaussian},
4 seeds x 4 prompts (n=16 per cell), reusing pass-1 directions
(`output/steering/*/20261022_164737/fit/directions.pt`). Generate configs:
`config/steering/{ukiyo_e,fauvism}_pass2.yaml`. Score tables under
`output/steering/*/20261022_164737/scores_*/`.

## Fauvism (VERIFIED)

| variant | dP(Fauvism) | dCLIP-T |
|---|---|---|
| SA/MW/true/a=4 | +0.180±0.031 | -0.013±0.007 |
| SA/DDIM/true/a=4 | +0.091±0.016 | -0.015±0.006 |
| SA/MW/true/a=2 | +0.027±0.008 | -0.002±0.005 |
| SA/DDIM/true/a=2 | +0.015±0.005 | +0.001±0.003 |
| swap / gaussian (all) | -0.001..0.000 | -0.017..+0.007 |

- Concept-specificity: both controls are flat zero at every setting; the
  steering effect comes from the concept difference, not from perturbation.
- Moonwalker beats DDIM: paired per-(prompt, seed) difference at a=4 is
  +0.090±0.024 (t=3.68, MW wins 14/16 pairs); same ordering at a=2.
  First measured task where inversion quality moves a downstream metric.

## Ukiyo-e (VERIFIED weak)

Best cell SA/MW/true/a=4: dP +0.011±0.005 with dCLIP-T -0.046; a=2 is inert.
Ordering MW > DDIM persists but the absolute effect is tiny; generations drift
to woodblock linework without hitting the CLIP class. Operating point (lower
alpha / step window) or a texture-sensitive metric needed before judging.

## Next steps

1. Ukiyo-e: alpha in {1.5, 2.5, 3} and step-window ablation (e.g. calls 1-25).
2. Text-free metrics: DINOv2 linear probe on WikiArt + CSD similarity to
   held-out positives (also unlocks truly non-promptable concepts).
3. E0 activation fidelity (true trajectory vs forward diffusion vs DDIM vs MW)
   to mechanistically explain the MW advantage.
4. Prompt-based CASteer reference arm for Fauvism (upper bound / comparison).
