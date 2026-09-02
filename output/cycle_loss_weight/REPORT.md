# Real weight of the cycle loss vs L_inv — SD1.4 DDIM, cfg=1

Measured, not derived. `CompVis/stable-diffusion-v1-4`, cfg=1.0, fp32, seed 42, 150 transitions over 3 prompts per grid length. Commit `e575d93f`.

Figure: `output/cycle_loss_weight/plots/cycle_loss_weight_sd14_20260902_150849.png` (SVG alongside).

## Headline

**At `lambda_cycle=1.0` on DDIM-50 the cycle term contributes 0.477% of L_inv — 1 part in 210.**

| quantity | value |
|---|---|
| mean `L_cycle / L_inv` | 4.7722e-03  (0.477%) |
| min / max over grid | 1.229e-04 / 1.279e-02 |
| `lambda_cycle` for parity | **210** |
| mean L_inv | 1.0259e-03 |
| mean L_cycle | 3.8036e-06 |
| mean B^2 | 5.1296e-03 |

## Grid-length sweep (panel f)

The weight is set by the step size, so it grows as the grid coarsens:

| DDIM steps | mean B^2 | mean ratio | lambda_cycle for parity |
|---|---|---|---|
| 10 | 1.2936e-01 | 8.192% | 12 |
| 20 | 3.2371e-02 | 2.682% | 37 |
| 50 | 5.1296e-03 | 0.477% | 210 |
| 100 | 1.2773e-03 | 0.123% | 815 |

At 50 steps (what `pnp_inversion` uses) the term is negligible. At 10 steps it is 8.2% and no longer ignorable.

## What this verifies

- **`L_cycle = B^2 ||e-v||^2` exactly.** Affine reconstruction of the DDIM step asserted against `scheduler.step()` at every timestep (tol 2e-3, passed).
- **`z_hat - z_i == -(B/A)(e-u)`** — max relative error 1.11e-03.
- **`||e-v|| ~ ||e-u||`**: measured 0.9850 (panel c), so the entire relative weight is the prefactor `B^2`. Panel (a) shows measured ratio tracking `B^2`.
- **Fixed point is unique.** Contraction factor `|B/A|*||J||` = 0.0961 << 1 (mean effective ||J|| = 1.729, range 0.92-8.3). So `L_cycle=0` <=> correct inversion; the degeneracy concern is empirically dead (panels d, e).
- `clip_sample=False` in SD1.4's shipped scheduler config, so the step really is affine. If it were True, x0 clipping would break invertibility outright.

## Per-timestep, DDIM-50 (prompt 1)

| t | B^2 | L_inv | L_cycle | ratio | \|e-v\|/\|e-u\| | J_eff |
|---|---|---|---|---|---|---|
| 981 | 1.519e-02 | 6.7478e-07 | 8.5332e-09 | 1.265e-02 | 0.9124 | 0.975 |
| 861 | 1.036e-02 | 5.2742e-06 | 5.6783e-08 | 1.077e-02 | 1.0194 | 1.135 |
| 741 | 6.998e-03 | 2.0761e-05 | 1.4178e-07 | 6.829e-03 | 0.9878 | 0.922 |
| 621 | 4.751e-03 | 5.9913e-05 | 2.6733e-07 | 4.462e-03 | 0.9691 | 1.147 |
| 501 | 3.300e-03 | 1.2500e-04 | 4.2142e-07 | 3.371e-03 | 1.0108 | 1.813 |
| 381 | 2.391e-03 | 4.6266e-04 | 1.2405e-06 | 2.681e-03 | 1.0589 | 2.807 |
| 261 | 1.865e-03 | 8.7934e-04 | 1.7333e-06 | 1.971e-03 | 1.0281 | 2.086 |
| 141 | 1.739e-03 | 1.8478e-03 | 3.4209e-06 | 1.851e-03 | 1.0318 | 2.733 |
| 21 | 1.001e-02 | 1.8528e-02 | 1.0596e-04 | 5.719e-03 | 0.7557 | 3.909 |
| 1 | 1.474e-04 | 5.6261e-05 | 7.0694e-09 | 1.257e-04 | 0.9232 | 8.310 |

## Note

Measured at LoRA initialisation, where the adapter is exactly the identity so `eps_phi(z_{i+1},t_i) = eps_theta(z_{i+1},t_i)`. The ratio is scale-free — both terms carry the same residual `(e-u)` and shrink together as training proceeds — so it stays ~`B^2` throughout training, not just at step 0.

`B^2` is a property of the SD beta schedule (scaled_linear, 0.00085->0.012, 1000 steps) and is therefore the same on SD1.5 and SDXL at equal grid length.

Reproduce:

```bash
HF_HOME=... .venv/bin/python scripts/measure_cycle_loss_weight.py --device cuda:0 --num_ddim_steps 50
.venv/bin/python scripts/plot_cycle_loss_weight.py
```