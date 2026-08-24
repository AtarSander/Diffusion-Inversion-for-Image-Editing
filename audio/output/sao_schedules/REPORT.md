# Stable Audio: two schedules under a first-order sampler with an exact inverse (2026-08-24)

`src/inversion_lora/compare_schedules.py`, 100 steps, one trajectory, no CFG. The solver is
DPMSolver++ first order (`FirstOrderSolver` in `stable_audio.py`), whose reverse step is affine in
the sample, so its inverse is exact arithmetic. Verified bit-identical to diffusers'
`dpm_solver_first_order_update` at three indices, with the inverse recovering the sample to 1e-8.

## 1. The exact inverse makes the objective well-posed

Round trip from the last invertible latent back to the initial noise, relative L2:

| inversion | beta grid | cosine grid |
| --- | --- | --- |
| oracle (exact reverse-pass data predictions) | **0.000001** | **0.000002** |
| substitution, noisier step's timestep ("shifted") | 0.010054 | 0.047407 |
| substitution, cleaner latent's own timestep ("matched") | 0.035053 | **0.017890** |

Compare `DPMSolverMultistepInverseScheduler`, whose oracle sits at 0.0153 -- worse than its own
approximation (`output/sao_pairing/REPORT.md`). With the exact inverse the ceiling is perfect on
both grids: an adapter that predicts the teacher's data prediction exactly inverts exactly. That is
what makes the shifted-denoiser objective meaningful here at all.

The final step is excluded: both schedules end at `final_sigmas_type="zero"`, where the reverse step
is `x_t = 0 * x_s + B * D` and discards the sample, so it has no inverse.

## 2. Which substitution is correct depends on the parameterisation

The winner flips between grids, by 3.5x one way and 2.6x the other:

- **beta** is VP (`alpha_t = 1/sqrt(sigma^2+1)`), the input is not rescaled, so declaring the
  noisier timestep for a cleaner latent is harmless and buys the closer target -- the classic DDIM
  shifted-denoiser trick, and what AudioLDM2 does.
- **cosine** is EDM: the input is scaled by `c_in(sigma)` and the output read through
  `c_skip(sigma)`/`c_out(sigma)`. Declaring the noisier sigma for a cleaner latent mis-scales both
  ends, which costs more than the shift buys. The matched pairing wins.

So on Stable Audio's native schedule the correct objective is
`D_phi(x_{i+1}, t_{i+1}) ~= D_theta(x_i, t_i)`: input the cleaner latent **at its own timestep**,
target the teacher's data prediction at the noisier one. Our cached trajectories encode the other
convention, on the other grid.

## 3. The beta grid is unusable

| | beta | cosine |
| --- | --- | --- |
| decoded audio RMS / peak | 4.05 / **22.49** | 0.085 / 0.454 |
| data-prediction RMS across the grid | 10.5 -> 121.8 | 0.24 -> 0.94 |

Audio must live in +/-1 and a latent should be O(1). On the beta grid the DiT is queried at
timesteps 999..10 where it was trained on 0.99..0.19, and its data predictions come out ~100x too
large; the decode clips by 22x. Audible in `beta_ode_100steps.wav` against `cosine_ode_100steps.wav`.

This also retires an earlier reassurance: beta's shift gap read 0.11-0.31% relative only because its
denominator is inflated ~100x. In absolute terms the two grids are comparable.

## 4. The real target on the cosine grid

Per-step substitution error in data-prediction space, quartiles of the grid, noisiest first:

| quartile | data RMS | e_RMS | e_rel |
| --- | --- | --- | --- |
| q1 | 0.2428 | 0.0117 | 4.81% |
| q2 | 0.3524 | 0.0238 | 6.75% |
| q3 | 0.6365 | 0.0726 | 11.41% |
| q4 | 0.9350 | 0.0418 | 4.47% |

A 4.5-11.4% per-step error, accumulating to a 1.8% round-trip error, against an exact ceiling. That
is a real objective, and much larger than what the beta grid appeared to offer.

## Consequences

- The corrected setting is: cosine sigma grid, first-order ODE, matched-timestep pairing. The
  existing 79 GB trajectory dataset matches none of those three, so it needs regenerating.
- The editing code's Stable Audio `ddim` mode runs on the beta grid, so the SAO **DDIM** baselines
  in the benchmark tables are measured with this broken sampler and need re-running. Its `ddpm` and
  `sdedit` modes use the native cosine scheduler and are unaffected.
