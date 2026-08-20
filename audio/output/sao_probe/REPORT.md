# Stable Audio Open teacher probe (2026-08-20)

Ran before porting the inversion LoRA, to settle *which* SAO teacher the adapter would distil.
Code: `src/inversion_lora/probe_stable_audio.py`. Numbers: `probe_100steps.json`,
`probe_20steps.json`. Audio: `cosine_100steps.wav`, `beta_100steps.wav`.

One prompt, one trajectory, no CFG, 100 steps, 47 s window, latent [64, 1024], fp32, A5000.

## VERIFIED

1. **SAO's native scheduler is an SDE.** `CosineDPMSolverMultistepScheduler` hardcodes
   `sde-dpmsolver++` and draws Brownian noise per step from an unseeded
   `BrownianTreeNoiseSampler`. Two runs with the same latent seed give different audio (decoded
   RMS 0.090 vs 0.315). There is no deterministic ODE variant of the cosine scheduler in
   diffusers 0.32.2. DDIM-style inversion is therefore undefined on SAO's native sampler.
2. **The editing code's `ddim` mode silently changes the noise schedule.**
   `DPMSolverMultistepScheduler.from_config(cosine_config)` drops `sigma_min`/`sigma_max`/
   `sigma_schedule` (they are not `DPMSolverMultistepScheduler` arguments) and falls back to a
   linear-beta grid: sigma 157.4..0.047 instead of 500..0.3, and timesteps 999..10 instead of
   0.9987..0.1855. The DiT's time embedding takes `log(t)` of a value the model saw in (0, 1),
   so it is queried ~1000x outside its trained range. It still decodes plausible audio
   (RMS 0.073, peak 0.72) — listen to `beta_100steps.wav` before trusting it.

## Shift gap, deterministic beta grid (the DDIM baselines' teacher)

The quantity the LoRA removes: inverting the step into `x[i+1]` needs the teacher at the noisier
`x[i]`, and substitutes the teacher at `x[i+1]`. Quartiles over step index, noisiest first.

| quartile | t range   | teacher RMS | e_RMS  | e_rel | bias/e_RMS |
|----------|-----------|-------------|--------|-------|------------|
| q1       | 999..759  | 0.2120      | 0.0018 | 0.85% | 0.142      |
| q2       | 749..509  | 0.2285      | 0.0053 | 2.30% | 0.124      |
| q3       | 500..260  | 0.2975      | 0.0103 | 3.45% | 0.091      |
| q4       | 250..10   | 0.4415      | 0.0163 | 3.68% | 0.047      |

There is a gap to close, and it grows toward the clean end (4.3x q4/q1 in e_RMS) — the same
direction as AudioLDM2, but far flatter there (AudioLDM2's MSE ratio was ~300x).

## Not measured

The cosine-grid rows in the JSON are computed over SDE trajectories, so they mix the Brownian
noise into the "gap" and are **not** a shift gap. Measuring one on the native schedule requires
writing a deterministic ODE sampler for it (~30 lines: dpmsolver++ update with the cosine
preconditioning).

## Wiring facts for the port

- Latent length is always `transformer.config.sample_size` = 1024 regardless of the requested
  duration; duration enters only through the timing conditioning and the final waveform crop.
  So storage is fixed at ~53 MB per fp32 trajectory at 100 steps (26.5 MB latents + 26.2 MB
  targets).
- The DiT accepts per-example timesteps: pass `[B]` rather than the pipeline's `t.unsqueeze(0)`.
  Conditioning (`text_audio`, `global_states`) expands over the batch; the rotary embedding is
  shared.
- The cosine scheduler's `scale_model_input` is **not** a no-op (`c_in = 1/sqrt(sigma^2 + 1)`),
  unlike DDIM/DPMSolverMultistep. A shifted forward must be preconditioned with the sigma of the
  timestep it claims to be at. Getting this wrong inflates the measured gap ~700x.
