# Shift gap vs classifier-free guidance — paired probe

**Date:** 2026-08-24
**Question:** does the DDIM shifted-denoiser gap — the error the inversion LoRA is trained to
close — grow with the guidance scale the trajectory was generated at?
**Method:** two probe sets of 8 trajectories, 200-step DDIM, **same prompts and same seed**, one at
`w=1.0` and one at `w=2.5`, both storing the conditional and unconditional branches
(`save_uncond_target: true`). Pure inference: for each transition, evaluate the CFG-combined
epsilon at the *cleaner* latent under the *noisier* timestep and compare against the combination
that actually advanced the stored trajectory. No training, no editing benchmark.

Reproduce:

```bash
uv run --python .venv/bin/python src/inversion_lora/generate_trajectories.py \
  --config-name generate_trajectories_cfg_probe device=cuda:N guidance_scale=<W>
uv run --python .venv/bin/python src/inversion_lora/measure_shift_gap.py \
  --data_root $LORAINV_DATA_ROOT/audioldm2_trajectories_cfgprobe_g<W> --guidance_scale <W>
```

## Result: the gap scales ~3x from w=1.0 to w=2.5

| band | w=1.0 | w=2.5 | ratio |
|---|---|---|---|
| **all transitions** | 4.060e-05 | **1.209e-04** | **2.98x** |
| q100_075 (noisiest) | 4.122e-07 | 8.928e-07 | 2.17x |
| q075_050 | 2.734e-06 | 8.085e-06 | 2.96x |
| q050_025 | 1.685e-05 | 2.479e-05 | 1.47x |
| q025_000 (cleanest) | 1.424e-04 | 4.497e-04 | 3.16x |

The band that dominates the loss, and the only one the `q4` runs train on, grows the most (3.16x).

## The combination is a minor term; the path is what matters

| w | combined | cond_only | difference |
|---|---|---|---|
| 1.0 | 4.060e-05 | 4.060e-05 | +0.0% |
| 2.5 | 1.209e-04 | 1.134e-04 | +6.6% |

`combined` is the gap against the CFG-combined target (what a pair-branch loss must close);
`cond_only` is the gap against the conditional target (what the current loss closes). At w=1 they
are identical to every digit, which is the correctness check on the measurement: at w=1 the
combination *is* the conditional epsilon. At w=2.5 they differ by only 6.6%.

So **generating the trajectories under guidance is the change that matters, not the choice of
target within a set.** The pair-branch storage is still required for correctness, because at
w != 1 the generator advances the latent with the combination while `target_eps` holds the
conditional branch alone -- without the unconditional branch the labels stop being exact for the
step they describe, which is the property that makes this dataset usable.

## Why this is a mechanism for the null result

Every adapter to date trained at `guidance_scale: 1.0` while the benchmark inverts at
`cfg_src: 3.0`. The adapter has therefore been solving a problem ~3x smaller than the one it is
deployed on (larger still at 3.0, extrapolating).

Concretely: at w=1 the best adapter closes 92.7% of a 4.06e-05 gap, leaving ~3e-06. At w=2.5 the
gap is 1.21e-04, so even if the learned correction transferred perfectly *in shape*, it is
calibrated to a 3x smaller error and would leave ~8e-05 uncorrected -- **twice the entire gap it
was trained to fix.** That is a quantitative reason the training objective can be driven to 92.7%
while the benchmark does not move.

## Sanity checks

- `combined == cond_only` at w=1.0 to the last digit (4.059764614794403e-05).
- The w=1.0 gap of 4.06e-05 sits next to the established LoRA-disabled loss of 4.83e-05, measured
  on the full validation split rather than 8 probe prompts. Same quantity, right magnitude.
- Band ratio q025_000 / q100_075 is 346x at w=1.0, matching the documented ~300x.

## What this does and does not establish

**Does:** the quantity the adapter corrects is a function of guidance, and the training regime has
never matched the deployment regime on that axis. It also lands where the ceiling argument is
weakest -- "DDPM-inversion is exact and traces the same front" bounded this whole direction, but
guidance dilutes that exactness, so the guided inversion error is the one quantity none of the six
negatives measured.

**Does not:** show that closing the guided gap moves any editing metric. Six measurements say
inversion fidelity does not, and this probe supplies a mechanism for the decoupling, not a refutation
of it. The dose-response on the benchmark is currently *inverted*, so a better-fitting adapter is
if anything predicted to hurt.

## Next step

Regenerate the full trajectory set at `guidance_scale: 2.5` with `save_uncond_target: true`, and
train with the pair-branch loss. Costs: 2x UNet forwards in generation (the unconditional pass
never runs at w=1, since `needs_uncond` is False), 2x target storage, and 2x forwards per training
step because the student must evaluate both branches. Storing both branches keeps the loss w a
config knob, so the training guidance can be swept without regenerating again.
