# Experiment log (most recent first)

## 2026-10-25 — CSD rescoring: all 8 styles move; MW wins on hard styles

- Hypothesis: paper's CSD style-similarity metric (to the 32 actual reference
  paintings) is fairer than CLIP/probe for image-derived steering.
- Method: ported official CSD ViT-L (tomg-group-umd/CSD-ViT-L) onto HF
  CLIPVisionModel (`diff_inversion/steering/csd.py`, strict load, smoke
  pos-pos 0.33 vs pos-neg 0.13); `score.py --csd_refs` + aggregate rerun.
- Result: every style gains dCSD +0.13..+0.45 (36–98% of ref-ref ceiling
  headroom) — probe "failures" (color_field, early_ren, ukiyo_e) move in style
  but not class membership → boundary restated: appearance transfers, class
  membership needs global change. Paired MW−DDIM dCSD: ukiyo_e +0.015 t=3.69
  (17/20), color_field t=2.01, cubism t=1.90; easy styles slightly favor DDIM.
  MW edge concentrates where inversion is hard. `SWEEP_RESULTS.md` updated.
- Next: n∈{4,8,16,32} source ablation; DDIM recon error vs MW−DDIM dCSD
  correlate; controls under CSD; spatial injection for class gap.

## 2026-10-24 — 8-style sweep: boundary is local-vs-global; MW ≈ DDIM at n=32

- Hypothesis: statistical styles (palette/texture) steer, structural styles
  (line/composition) fail; MW>DDIM holds across styles.
- Method: 8 styles × (32 pos + 32 mixed-neg fresh WikiArt images), DDIM+MW
  directions, 20 prompts × α 0.5–3.0, SA + renorm, probe + CLIP scoring
  (`scripts/steering_sweep_aggregate.py`, runs `output/steering/sweep_*/`).
- Result: works (dP_probe +0.12..+0.15): fauvism, pointillism, impressionism,
  cubism, pop_art; fails/weak: color_field (+0.015), early_renaissance
  (+0.034), ukiyo_e (+0.048). Grouping falsified — operative axis is local
  appearance statistics vs global composition/content change. MW≈DDIM at 32
  sources (only pointillism α=3 t=2.16; impressionism trends DDIM); contrasts
  with MW>DDIM t=3.68 at 16 sources. Details:
  `output/steering/SWEEP_RESULTS.md`.
- Next: source-count ablation n∈{4,8,16,32} (is MW's edge sample-efficiency?);
  controls for the 4 new working styles; spatial/attention-map injection for
  structural styles.

## 2026-10-23 — WikiArt style probe; text-free metric rehabilitates ukiyo-e

- Hypothesis: CLIP zero-shot undersells ukiyo-e steering due to prompt priors;
  a text-free classifier gives a fairer verdict. No usable pretrained WikiArt
  style classifier exists on HF (artist/genre heads or 9 styles only).
- Method: `diff_inversion/steering/probe.py` — streamed WikiArt sample (cap
  200/style), frozen dinov2-base CLS features, sklearn logistic regression;
  27-way val acc 0.508 (Ukiyo_e 0.90, Fauvism 0.55). `score.py --probe_path`
  adds P_probe columns. Probe at `output/steering/probe/wikiart_dinov2_probe.pt`.
- Result (fine grids, renorm, n=5): ukiyo-e dP_probe +0.097 (MW a=2) / +0.092
  (DDIM a=2) — ~4x the CLIP signal; fauvism peaks MW a=2.5 dP_probe +0.234 vs
  DDIM +0.192, MW > DDIM at every matched alpha 1.5-2.5. Also fixed the
  Mannerism_Late_Renaissance label in the CLIP prompt list.
- Next: controls under the probe metric; step-window/layer ablations;
  forward-diffusion baseline arm; E0 activation fidelity.

## 2026-10-22 — Pass 2: controls confirm concept-specific steering; MW > DDIM

- Hypothesis: pass-1 Fauvism effect is concept-specific (not perturbation) and
  the MW-over-DDIM gap survives seeds + controls.
- Method: SA-only, alpha {2,4}, controls {true, swap, gaussian}, 4 seeds x 4
  prompts, pass-1 directions reused.
- Result: controls dead flat (dP -0.001..0.000); true directions steer
  (MW +0.180±0.031 vs DDIM +0.091±0.016 at a=4); paired MW-DDIM +0.090±0.024,
  t=3.68, 14/16 wins. Ukiyo-e stays weak (+0.011 max, CLIP-T cost).
  Details: `output/steering/PASS2_RESULTS.md`.
- Next: ukiyo-e operating point; DINOv2/CSD metrics; E0 activation fidelity;
  prompt-based CASteer reference arm.

## 2026-10-22 — Image-derived steering pass 1 (WikiArt Ukiyo-e, Fauvism)

- Hypothesis: steering vectors can be fit from real images alone (inverted
  trajectories, unconditional activation reads), with SA carrying the signal
  and inversion quality (DDIM vs Moonwalker) affecting direction quality.
- Method: 16 pos vs 16 mixed-neg WikiArt images per concept; DDIM + MW
  inversion at 50 steps; CASteer addition on last-36 SDXL blocks; CLIP
  zero-shot P(style) + CLIP-T scorer (`diff_inversion/steering/score.py`).
- Result: Fauvism steers (SA/MW/a=4: P 0.222 vs 0.001 base, content intact);
  SA >> block >> CA (CA inert, as predicted for null-text reads); MW > DDIM
  3x on Fauvism CLIP score (uncontrolled, n=4); Ukiyo-e drifts to woodblock
  linework but overshoots at a=4. Details: `output/steering/PASS1_RESULTS.md`.
- Next: pass 2 controls (swap/gaussian, 4 seeds) running on GPUs 4-5;
  ukiyo-e operating point; DINOv2/CSD metrics; E0 activation fidelity.
