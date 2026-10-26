# Experiment log (most recent first)

## 2026-10-26 — up0a01 arm: attn.0+1 jointly ≈ up0; best Young setup; only subset with any eyeglasses-add. Triangle-size dataset extracted+fit

- Hypothesis: steering up_blocks.0.attentions.{0,1} jointly recovers up0's
  effect (a2 contributes nothing) and may be superadditive where single
  attentions fail.
- Method: 10th layer setup `up0a01` (`--layers up_blocks.0.attentions.0,
  up_blocks.0.attentions.1`), same grid/vectors as the ablation (3 jobs,
  GPUs 4–6). Aggregation rerun → `output/steering/celeba_summary/
  20261026_193451_celeba_layers_summary.md` (+ per-attr PNGs).
- Result: (1) Smiling: +0.43 at α=+3 — 80% of up0 (+0.54), above up0a0
  (+0.37), id_sim same as up0 (0.16); joint ≈ sum of parts, no identity gain.
  (2) Young: best setup overall — +0.29 at +3 (up0 +0.18, last36 +0.08)
  with BETTER identity than up0 (0.25 vs 0.19); aging at −1 ties up0 (−0.37).
  (3) Eyeglasses: the only subset with a positive add signal (+0.13 at +3;
  a0 alone −0.03, a1 alone −0.01 → superadditive), though still well below
  last36 (+0.31); removal −0.18 ≈ up0. (4) MW vs DDIM: within-noise again.
- Interpretation: attn.0+attn.1 of up_blocks.0 is the working pair — a2 is
  dead weight whose steering only costs identity. For Young it strictly
  dominates everything; for object insertion it helps but coarse blocks are
  still required.
- Also: triangle-size synthetic dataset (64+64 matched pairs, only size
  differs) extracted+fit on GPU 7 → `output/steering/triangles_size/
  20261026_172253/{extract/features.pt (12 GB, per-example pos/neg
  activations per SA layer × 50 steps, both methods), fit/directions.pt
  (287 MB)}`. Ready for direction analysis / generation.
- Next: use triangle per-example activations (user-directed); up0a01 ×
  start_call>1 as the identity-preserving recipe candidate.

## 2026-10-26 — CelebA layer ablation: up_blocks.0 is the attribute locus; eyeglasses-add is the exception

- Hypothesis: attribute directions live in a subset of the 36 steered blocks;
  steering fewer blocks preserves identity better.
- Method: 9 layer setups (mid, up0, up1, up0a{0,1,2}, up1a{0,1,2}) × 3
  attributes × same α grid, reusing the fitted 20261026_024550 directions via
  `generate --layers` (27 jobs, GPUs 0–7). Aggregation
  `scripts/analyze_celeba_layers.py` → `output/steering/celeba_summary/
  20261026_152927_celeba_layers_summary.md` (+ per-attr PNGs).
- Result: (1) Smiling: up0 beats last36 (dP_probe +0.54 vs +0.31 at α=+3);
  effect concentrates in up0a0 (+0.37 at 2× the id_sim, 0.26 vs 0.14); decays
  a0→a1→a2; mid/up1 inert. (2) Young: same locus — up0 beats last36 both ways
  (aging α=−3 −0.57 vs −0.29; rejuvenation +0.18 vs +0.08 at +3), up0a0/up0a1
  carry it; up1 is wrong-sign (ages at +α, −0.25). (3) Eyeglasses breaks the
  pattern: NO subset adds glasses (all ≈0 at +3 vs last36 +0.31 — needs
  up0+up1 jointly); removal (α=−1 −0.20) is reproduced by up0 alone. (4)
  Identity: every subset preserves id_sim better than last36; mid barely alters
  the image. (5) MW vs DDIM within setups: mixed, no consistent edge (Young up0
  +3: MW +0.25 vs DDIM +0.10; Smiling up0: DDIM +0.60 vs MW +0.49).
- Interpretation: localized semantic attributes (expression, age) are encoded
  early in up_blocks.0 and over-steering extra blocks only destroys identity;
  adding an object (glasses) recruits coarse+fine blocks jointly.
- Next: identity-preserving recipe = up0a0 (+ start_call>1); conditioned /
  orthogonalized directions vs CelebA confounds; Linear-AcT arm.

## 2026-10-26 — CelebA face attributes: direction found, identity not preserved

- Hypothesis: InterfaceGAN-style attribute directions (Smiling/Eyeglasses/Young)
  fit from 32+32 covariate-matched CelebA-HQ faces via inversion steer SDXL
  generations; MW > DDIM where inversion is hard.
- Method: full pipeline per attribute (extract→fit→generate→score_faces) on
  GPUs 4/5/6, 20 neutral portrait prompts × seed 42 × α∈[−3,+3]. Metrics:
  DINOv2 40-attr probe on MTCNN crops (primary), CLIP zero-shot P(attr),
  FaceNet id_sim to base, CLIP-T. Runs `output/steering/celeba_<attr>/20261026_024550`,
  summary `output/steering/celeba_summary/20261026_055909_celeba_summary.md`.
- Result: (1) Addition works where base rate leaves headroom: Smiling dP_probe
  +0.31/+0.32 (DDIM/MW) and dP_clip +0.26/+0.31 at α=+3, monotone; Eyeglasses
  only kicks in at α=+3 (+0.37/+0.26); Young+ is weak/non-monotone (base 0.56).
  Erasure mirrors headroom: aging (Young α=−1) dP_probe −0.27/−0.38, Eyeglasses
  α=−1 −0.23/−0.18; Smiling− floor-limited (base 0.12). (2) Identity is NOT
  preserved: id_sim 0.1–0.45 at every α (same-person ≈0.7) — steering from
  call 1 at 36 blocks re-lays-out the scene; this is distribution-level
  steering, not editing. (3) Face-miss asymmetry: negative α degrades images
  into haze (found 5–50% vs 60–90% at positive α); base itself 65% (wide-scene
  prompts hide faces). (4) Off-target drift at α=+2 = CelebA entanglement:
  Smiling+ → less Male (−0.14)/less Young (−0.13); Young+ → less Male (−0.23),
  fewer Eyeglasses (−0.18). (5) MW vs DDIM mixed: equal on Smiling, paired MW−DDIM
  p_probe +0.082±0.022 (n=29) on Young+ (hardest/global attribute — echoes the
  hard-style CSD edge), −0.027±0.044 ns on Eyeglasses.
- Next: identity-preserving variant (start_call>1, fewer blocks, or lower α on
  later steps); portrait-only prompts to kill base misses; per-α matched-pair
  controls (swap/gaussian) before claiming attribute specificity.

## 2026-10-26 — All-layers ablation: steering down/mid blocks buys nothing

- Hypothesis: the ukiyo_e/color_field class-membership gap (style moves, probe
  doesn't) comes from only steering up blocks (last36); adding down/mid (all 70
  SA blocks) should move global composition.
- Method: `generate --config config/steering/all_layers_{ukiyo_e,fauvism}.yaml`
  reusing the fitted sweep directions, same 20 prompts × α grid; scored with
  CSD + probe. Comparison: `output/steering/sweep_summary/20261026_014853_all_vs_last36.md`.
- Result: falsified. dCSD unchanged (ukiyo_e) or slightly worse (fauvism at
  α≥1.5); ukiyo_e dP_probe *drops* (MW α=1.5 +0.048→+0.022); ukiyo_e CLIP-T
  degrades 2× faster at α=3 (−0.034). Fauvism gains some probe/CLIP-T at α=3
  but loses CSD. Side plots: DDIM-vs-MW SV cosine ≈0.93–0.99 across all 70
  blocks (cross-style ≈0), so down/mid directions are well-defined — they just
  don't help. last36 stays the default.
- Next: class-membership gap likely needs spatial/attention-map injection, not
  more layers. CelebA face-attribute steering planned (smile/age/glasses,
  symmetric α for erasure, DINOv2 attribute probe + ArcFace identity).

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
