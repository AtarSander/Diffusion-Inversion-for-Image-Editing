# Experiment log (most recent first)

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
