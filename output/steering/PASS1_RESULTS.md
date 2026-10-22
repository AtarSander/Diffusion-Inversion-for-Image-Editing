# Image-derived steering, pass 1 (2026-10-22)

Steering vectors computed from real WikiArt images only (no concept prompts):
16 positives vs 16 style-capped mixed negatives per concept, DDIM vs Moonwalker
(MW) inversion at 50 steps, activations read unconditionally (empty prompt) at
every saved inversion state, CA/SA/block representations, CASteer addition on
the last 36 SDXL blocks, guidance 5.0, 4 neutral prompts x 1 seed,
alpha in {1, 2, 4, 8}. Code at commit b9b122f; runs under
`output/steering/{ukiyo_e,fauvism}/20261022_164737/`.

Metric: CLIP zero-shot P(target style) over the 27 WikiArt style prompts and
CLIP-T prompt adherence, each paired against the same-(prompt, seed) base
image. Full tables: `*/20261022_164737/scores_*/*_clip_scores.md`.

## Findings (VERIFIED on scores + visual inspection; n=4, no controls yet)

1. Image-derived steering works. Fauvism SA/MW/alpha=4: P(Fauvism)
   0.222±0.098 vs base 0.001, CLIP-T -0.011 (content preserved). Images show
   flat saturated planes, blue contours, simplified forms.
2. SA is the operative representation; CA is inert (alpha=8 leaves CLIP-T at
   +0.02 and P unmoved), matching the prediction that null-text cross-attention
   output carries little image-derived concept signal. block is between.
3. MW directions beat DDIM at matched settings on Fauvism CLIP score
   (0.222 vs 0.067 at SA/alpha=4); visually both steer, MW somewhat harder.
   NOT yet a claim: n=4, no swap/gaussian controls, single seed.
4. Ukiyo-e steers toward woodblock line-engraving texture but misses the CLIP
   class: at SA/alpha=4 composition collapses into allover linework
   (P(Ukiyo_e) +0.006 max, CLIP-T -0.03..-0.12). Its operating strength is
   lower/narrower than Fauvism's; concept difficulty is style-dependent.
5. alpha=8 overshoots everywhere (CLIP-T -0.08..-0.12).

## ASSUMED (to verify in pass 2+)

- MW>DDIM gap survives swap/gaussian controls and more seeds (pass 2 running:
  SA-only, alpha {2,4}, 4 seeds x 4 prompts, all three controls).
- CLIP zero-shot is a fair ranking metric here; it requires a promptable style,
  so Ukiyo-e's near-zero P may partly be metric failure (planned: DINOv2 linear
  probe + CSD similarity to held-out positives).

## Next steps

1. Pass 2 (running): controls + seeds at SA alpha {2,4}.
2. Ukiyo-e operating point: finer alpha (1.5-3) and/or step windowing
   (steer early/mid calls only).
3. Text-free metrics (DINOv2 probe, CSD) for the non-promptable story.
4. E0 activation-fidelity experiment on generated images (true trajectory vs
   forward diffusion vs DDIM vs MW) to ground the OOD-activations claim.
