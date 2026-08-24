# Is the inversion loss correctly defined on Stable Audio Open? (2026-08-24)

No. `src/inversion_lora/verify_inversion_pairing.py`, one trajectory, `beta` grid, no CFG.

## VERIFIED

### 1. The inverse solver queries timesteps the training pairs never contain

`DPMSolverMultistepScheduler.set_timesteps(N)` builds the reverse grid {999, 989, ..., 10} while
`DPMSolverMultistepInverseScheduler.set_timesteps(N)` builds {0, 10, ..., 989}: offset by exactly
one grid step. The training pair for transition `i` is `(trajectory[i+1], timesteps[i]) ->
outputs[i]`, so at every one of the 100 steps the adapter is asked for a timestep one step cleaner
than the one it was trained on. 100/100 mismatched at N=100; 20/20 at N=20.

### 2. The target does not control the round-trip error

Feeding the inverse solver the *exact* model outputs the reverse pass used -- which is what a
perfect adapter would predict -- recovers the initial noise **worse** than the ordinary
approximation does:

| inversion | relative L2 to the true initial noise, N=100 | N=20 |
| --- | --- | --- |
| oracle, exact reverse-pass outputs | 0.0153 | 0.0639 |
| plain, solver timesteps (what the editing code runs) | 0.0080 | 0.0399 |
| plain, training-pair timesteps | **0.0037** | **0.0269** |

So `DPMSolverMultistepInverseScheduler` is not the algebraic inverse of
`DPMSolverMultistepScheduler` on the same grid, and the shifted-denoiser "shift gap" is not the
error term that limits this inversion. An adapter that learned the objective perfectly would make
inversion about 1.9x worse than not trying. Per-step confirmation: one oracle step from the clean
latent lands 0.0048 away from the reverse trajectory's neighbour instead of 0.

### 3. A free 2.2x improvement is available with no adapter at all

Querying the model at the reverse grid's timestep instead of the inverse scheduler's cuts the
inversion error from 0.0080 to 0.0037 (0.0399 -> 0.0269 at N=20). One index change in
`ddm_inversion/ddim_inversion.py` for the Stable Audio path.

## What this invalidates

The Stable Audio LoRA results in `output/sao_lora/20260824_103855/` were measured against this
mis-specified objective, so they do **not** support "inversion fidelity does not limit editing on
Stable Audio". The adapter was not optimising inversion fidelity. The +1.3 dB reconstruction gain
is real as measured but uninterpretable as evidence about the objective.

The AudioLDM2 result is unaffected: there the reverse and inverse passes share one DDIM grid and
`next_step` is the exact algebraic inverse, so the shift gap is the error term the adapter targets.

Separately, listening to `output/sao_probe/beta_100steps.wav` against `cosine_100steps.wav`: the
beta-grid teacher sounds clearly degraded, which is the schedule mismatch in
`output/sao_probe/REPORT.md` showing up audibly. Two independent defects in the same teacher.
