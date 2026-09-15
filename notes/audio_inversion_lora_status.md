# Audio Inversion LoRA — Status & Open Items

Porting the shifted-denoiser inversion LoRA (see [inversion_methods.md](inversion_methods.md))
from SDXL/SD1.5 to **AudioLDM2**, to be compared against the DDPM-inversion / DDIM / SDEdit
baselines in `audio/editing/` on the MedleyMD editing benchmark.

Most recent first. Keep this file current — it is the handover doc between sessions.

---

## 2026-09-15 (later) — BUILD: pair-branch loss and its inference path, on the guided dataset

**Why.** `output/guided_inversion/` showed the shared-CFG adapter and its no-LoRA control
collapsing *together* at cfg_src=3.5, so guided inversion is what breaks, not the adapter. One
concrete mechanism survives that: with a shared adapter, whatever error it introduces enters the
edit through `w * (cond - uncond)` and is amplified by w. Pair-branch removes that coupling by
giving the empty prompt its own adapter.

**The loss** (training config 36, `saocos_cfg35pair_r8_a4_lr5e-5`):

```
L = ||D_phi(x_{i+1}, c)     - D_theta(x_i, c)||^2
  + ||D_nu (x_{i+1}, empty) - D_theta(x_i, empty)||^2
```

Each branch against its own frozen teacher, **no w in the loss at all**. The guided dataset is
used because its trajectories visit the latents guided inversion actually reaches, and because it
already caches both branches — not because the loss needs a guidance value.

In the taxonomy we were comparing: the earlier cfg35 run is **Shared CFG Loss** (one adapter,
guided combination on both sides), which is also what AudioLDM2 config 28 does despite being
named "pair-branch" in its comment. True Pair Branch did not exist before this.

**Two traps, both now guarded by tests.**

1. PEFT's `set_adapter` FREEZES the adapters it deactivates. A naive per-branch switch leaves the
   unconditional adapter with no gradient and it silently never trains, while parameter counts and
   loss curves look healthy. `requires_grad` is restored after every switch;
   `tests/test_pair_branch_adapters.py` asserts both adapters receive gradient AND that
   `set_adapter` alone would starve one, so the guard cannot be deleted unnoticed.
2. Adapters must be injected BEFORE the base trainer reads trainable parameters and builds the
   optimiser, hence the `inject_extra_adapters` hook. Confirmed by the count doubling to
   8,257,536 across 768 tensors.

**Inference cannot use the merge trick.** `attach_inversion_lora` folds the adapter into the base
weights for speed, but a guided step needs *different* weights for its two calls, and
merging/unmerging twice per step across every module costs more than the side branch does. So
`attach_pair_inversion_lora` keeps both adapters unfused and routes with `set_adapter`; these runs
are therefore slower than the merged single-adapter arms, which matters for wall-clock comparisons
but not for NFE. Detection is automatic on the sibling `<stem>_uncond.pt`, so single-adapter
checkpoints keep the cheaper path.

**Also added:** the w=1 adapter's missing cfg_tar=7.0 cells at step 2000
(`sao_w1_cfgtar7_configs.sh`), so the two adapters can be compared at both target guidances rather
than only at 3.5; and the no-LoRA control at cfg_src=3.5, which is what produced the correction
above.

**Standing decision:** train to 3000 steps with checkpoints every 500. The ladder put saturation
at 2000 but only sampled every 2000, so 3000 keeps a margin past the knee and the finer cadence
will show whether the knee is actually below 2000.

**Next:** score config 36 at cfg_src 1.0 and 3.5 when it lands. If the pair-branch adapter also
collapses at cfg_src=3.5, the guidance-amplification mechanism is dead and the remaining
explanation is that guided ODE inversion is unstable on this solver independently of any adapter
— which is its own diagnostic (round-trip error at cfg_src 1.0/2.0/3.5, no adapter).

**Still owed, sixth attempt:** the reconstruction ladder, 0/12.

---

## 2026-09-15 — CORRECTION: experiment 1 is untested, not refuted. Guided inversion is broken.

I previously wrote that the CFG-aware adapter fails and therefore "the w=1/w=3.5 mismatch is not
what was holding the adapter back". That was over-claimed. The no-LoRA control at cfg_src=3.5,
which had never been run, collapses exactly the same way.

cfg_src=3.5, cfg_tar=3.5, real-audio hparam split, 115 edits:

| tstart | no LoRA LPAPS / CLAP | CFG adapter LPAPS / CLAP | cfg_src=1.0 reference |
|---|---|---|---|
| 50 | 5.187 / 0.220 | 5.216 / 0.211 | 4.500 / 0.309 |
| 75 | 5.974 / 0.160 | 6.039 / 0.129 | 5.342 / 0.316 |
| 99 | 6.078 / 0.141 | 6.111 / 0.119 | 5.431 / 0.310 |

**The two arms are within 0.07 LPAPS of each other everywhere**, while both sit ~0.7 LPAPS and
~0.16 CLAP worse than unguided inversion. It is guided inversion that breaks, not the adapter.
The matched configuration the experiment was built to test cannot be run at all on this solver,
so the hypothesis stands untested — the earlier "experiment 1 fails" entry should be read with
that correction.

cfg_tar=7.0 partially rescues alignment for both arms (adapter CLAP 0.129 -> 0.285 at t75) but
not preservation, and the adapter still never beats its control. Raising target guidance
compensates for a degraded latent rather than repairing it.

**What this does and does not change.** The ceiling result is untouched: the four nulls that
support it (cycle k=1, cycle k=2/3, 10x training, and the CFG adapter at cfg_src=1.0) all stand.
What changes is that "CFG-aware training does not help" is no longer one of them — it was never
tested in its matched configuration. Testing it needs guided ODE inversion to work first, which
is its own investigation: the failure signature (preservation AND alignment degrading together,
worsening with tstart) looks like a diverging latent rather than a bad operating point.

**Still owed:** the reconstruction ladder, 0/12, never submitted across four attempts. It remains
the measurement that would turn "editing is flat against training step" into "flat against
measured round-trip error".

---

## 2026-09-15 — RESULT: H2 is a clean null; the matched-NFE sweep restores ODEInv's tail advantage

**H2 (dense training data): null.** The dense991 adapter (baseline 2.692e-4, val 2.72e-5 = 89.9%
closed — same objective fit as the coarse adapter) reproduces the coarse adapter's editing effect
to the third decimal. Paired dense − coarse@4000 over the 8 cells: dLPAPS −0.0005..−0.0062 (three
cells p<0.05, all ~100x smaller than the adapter effect itself), every other metric ±0.001.
Dense − no-LoRA matches the coarse deltas cell for cell. So a verified-exact 10x-denser training
trajectory — per-step input shift 1.4e-4, ~1e-2 accumulated — buys ~0.003 LPAPS. Together with H1
(the full real-vs-generated shift buys ~0.07 LPAPS, ~4% of the front), the dose-response is
roughly proportional and small everywhere: **the input-distribution axis is not where the editing
ceiling lives.** Runs: `saocos_dense991_r8_a4_lr5e-5` cells vs their twins; script pattern in
output/gen_inputs. Do not spend further compute on trajectory quality without a new mechanism.

**Matched NFE at cfg 3.5–14 (48 new runs, `output/matched_nfe/20260914_191008`).** At equal
compute DDPM-inv does NOT catch the ODE tail: MuQ 0.286 vs ODEInv+LoRA 0.308 at LPAPS ~5.3; the
s100 near-merge was partly DDPM's per-point NFE advantage (2(100+t) vs 3t). Fronts split by
regime: DDPM the preserved end, ODEInv±LoRA the aligned end, SDEdit the extreme tail. The LoRA's
paired LPAPS gain persists at every guidance (−0.02..−0.08), shrinking slightly with cfg.

**H1 figure** (`output/gen_inputs/20260914_222605/`): fronts + paired deltas, real vs generated.
On generated inputs all fronts shift far left (easier preservation) but alignment ceilings DROP
(dir-MuLan max ~0.20 vs ~0.35 real), and DDPM-inv saturates earliest — exact inversion is worth
more on-manifold.

**Ops hardening shipped this session** (all on audio_edit): SPLIT/CFG_TARS-parameterized grids
with grid-owned `medleymd/stable_audio` subdir; eval split-leak fix; space-safe CFG_TARS through
sbatch (colon form); `submit_edit_eval.sh` one-command pair submission with retries;
`--skip_existing` support in the SAO edit driver (Fire rejected it only AFTER a full edit, so
retries tripled work and exit codes lied); SRC/ARM_KIND/PROBE forwarded explicitly;
`verify_trajectories.py` covers SAO plain + dense datasets.

---

## 2026-09-14 (evening) — RESULT: H3 mostly closes, H1 finds a real but small distribution gap

Scored runs: `output/all_methods/20260914_163319` (hparam, DDPM cfg 3.5–14) and
`20260914_163322` (genhparam). H2 (dense991) still training; dense-vs-coarse input gap measured
at rel mean 1.4e-4 / max 4.3e-4 per step — real signal, chain left running.

**H3: the MuQ/directional "ceiling" was mostly a guidance artifact.** DDPM keeps climbing past
cfg 7: MuQ 0.242 → 0.272 (10.5) → 0.288 (14, t99); mulan_dir 0.278 → 0.314 → 0.334. At matched
preservation the fronts now touch: DDPM cfg14/t50 (LPAPS 4.79, MuQ 0.275) slightly dominates
ODEInv+LoRA cfg7/t50 (4.88, 0.273). ODEInv keeps only the extreme-alignment tail: its best MuQ
0.306 / mulan_dir 0.349 (cfg7, t99, LPAPS 5.66) vs DDPM's 0.288 / 0.334 at LPAPS 5.25. CLAP
saturates for both (~0.345–0.349). NFE identical per DDPM row across cfg; DDPM spends more NFE
than ODEInv on this grid, so its residual shortfall at the tail is conservative.

**H1: the real-vs-generated gap exists, doubles the adapter's effect, and still does not change
the regime.** Paired LoRA@4000 − no-LoRA at cfg 3.5 on generated inputs: dLPAPS −0.138/−0.133/
−0.114 at t50/75/99 (all p<0.001) with CLAP now positive (+0.004..+0.0075, p<0.01 at t75/99) —
preservation and alignment improve together, ~2x the real-audio deltas (−0.062/−0.080/−0.077).
So training on generations does cost transfer to real audio — but even fully in-distribution the
adapter moves LPAPS by ~4% of the front. The editing ceiling is not a train/test artifact.
Side observation: on generated inputs ODEInv dominates DDPM-inv outright (DDPM saturates at
CLAP 0.269 / MuQ 0.175 where ODEInv reaches 0.30 / 0.25) — exact inversion is worth more when
the input actually lies on the model's trajectory manifold.

**Ops note.** Three eval-array failures before the numbers landed, both mine: the eval script's
`SPLIT=full` default leaked into the now-parameterized grid files (fixed: grid sees the
submission's SPLIT), and `LORA_EDITS_SUBDIR` was retyped as `stable_audio` instead of the doubled
`medleymd/stable_audio` (fixed: SAO grid files own the value). Edits were never affected.

---

## 2026-09-14 (later) — BUILT: three experiments (generated inputs / dense training data / DDPM high-cfg)

**H3 partial answer from disk first.** The all-methods plot pinned cfg_tar to 3.5, but the hparam
sweep also ran DDPM/SDEdit at 7.0. Pooled (`output/all_methods/20260914_134443/`): on **CLAP** DDPM
catches ODEInv at cfg 7.0 (0.335 @ LPAPS 4.67 vs LoRA's 0.336 @ 4.88) — but on **MuQ and both
directional metrics it does not**: DDPM tops out at MuQ 0.243 / mulan_dir 0.278 while ODEInv±LoRA
reach 0.306 / 0.349. `compare_all_methods.py` now pools every cfg_tar per arm, adds the mulan_dir
panel and takes `--split`. The open question is whether cfg_tar 10.5/14 closes that gap; note the
grid holds NFE fixed per method automatically (CFG costs 2 forwards/step at any scale != 1), and on
the s100 grid DDPM already spends 2*(100+t) forwards vs ODEInv's 3t, so a DDPM failure to reach the
ceiling is conservative.

**H1 (edit generations instead of real audio).** New `genhparam` split: 75 clips = one per unique
(track, source caption) pair of the 115 hparam rows, generated by seeded cosine-ODE at w=1 and 100
steps — exactly the LoRA's training distribution. Durations follow the real tracks capped at the
47.55 s window; **seeds 314159..314233 are reserved for eval inputs, never reuse for training**
(training used 42..1541). Committed CSV `captions_gpt5_genhparam.csv` pins names/durations/seeds;
`medley_audio_root(split)` points drivers/eval/lower-bound at `MEDLEY_GEN_AUDIO_DIR`. Grid files
take `SPLIT=`/`CFG_TARS=` from the submission.

**H2 (dense training data).** `generate_trajectories_stable_audio_dense.yaml`: 991 fine ODE steps
(991 = 10*99+1 so the fine sigma grid nests the 100-point coarse grid exactly; a plain 1000-point
grid shares only 10 points). Training pairs stay on the coarse grid: **naive every-10th subsampling
breaks the exact step relation** (ten fine steps != one coarse step — the same defect class as the
retracted 2026-08-24 objective), so each input is formed as one exact coarse step from the fine
state, `input_k = A y_k + B D(y_k)` — exact pair, dense-quality states, trainer untouched. The
dense-vs-coarse input gap is logged per sample and stored in meta; if it is ~0 the experiment is
dead before training. `verify_trajectories.py` now checks SAO datasets (plain + dense) with a
vectorized exact-step invariant. Training preset 36 (`saocos_dense991_r8_a4_lr5e-5`), edit grid
`sao_dense_sweep_configs.sh`. Costs ~10x generation (~4 h on 8 H100 shards), 118 GB — check space.

**Verified**: dense generate→verify→dataset pairing smoke-tested on GPU (2 samples); gen-inputs
smoke (2 clips, correct layout/sr/duration); grid files reproduce the old 24-row grid byte-for-byte
without overrides; 4 unit tests (`test_dense_trajectories.py`, `test_generate_eval_inputs.py`).

**Launch (WCSS, from `audio/`, `WCSS=editing/AudioEditingCode/code/slurm_scripts/wcss`):**

```bash
# H3: DDPM at cfg_tar 10.5/14 (8 edits ~3 h each + evals, chained)
EDIT=$(METHOD=ddpm CFG_TARS="10.5 14.0" SWEEP_CONFIGS=$WCSS/sao_methods_hparam_configs.sh \
  SCRIPT=edit_stableaudio_medleydb.py LORA_EDITS_SUBDIR=stable_audio \
  bash $WCSS/submit_lora_sweep.sh --array=0-7 | tail -1 | awk '{print $NF}')
ARM=lora METHOD=ddpm CFG_TARS="10.5 14.0" SWEEP_CONFIGS=$WCSS/sao_methods_hparam_configs.sh \
  LORA_EDITS_SUBDIR=stable_audio bash $WCSS/submit_eval.sh --array=0-7 --dependency=afterok:$EDIT

# H1: inputs + lower bound, then 16 edits (ddpm/sdedit 4 each, odeinv nolora+step4000 8) + evals
GEN=$(sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
  $WCSS/run_generate_eval_inputs.sh | awk '{print $NF}')
for M in ddpm sdedit; do
  E=$(METHOD=$M SPLIT=genhparam CFG_TARS=3.5 SWEEP_CONFIGS=$WCSS/sao_methods_hparam_configs.sh \
    SCRIPT=edit_stableaudio_medleydb.py LORA_EDITS_SUBDIR=stable_audio \
    bash $WCSS/submit_lora_sweep.sh --array=0-3 --dependency=afterok:$GEN | tail -1 | awk '{print $NF}')
  ARM=lora METHOD=$M SPLIT=genhparam CFG_TARS=3.5 SWEEP_CONFIGS=$WCSS/sao_methods_hparam_configs.sh \
    LORA_EDITS_SUBDIR=stable_audio bash $WCSS/submit_eval.sh --array=0-3 --dependency=afterok:$E
done
E=$(SPLIT=genhparam CFG_TARS=3.5 SWEEP_CONFIGS=$WCSS/sao_lora_sweep_configs.sh \
  SCRIPT=edit_stableaudio_medleydb.py LORA_EDITS_SUBDIR=stable_audio \
  bash $WCSS/submit_lora_sweep.sh --array=0-7 --dependency=afterok:$GEN | tail -1 | awk '{print $NF}')
ARM=lora SPLIT=genhparam CFG_TARS=3.5 SWEEP_CONFIGS=$WCSS/sao_lora_sweep_configs.sh \
  LORA_EDITS_SUBDIR=stable_audio bash $WCSS/submit_eval.sh --array=0-7 --dependency=afterok:$E

# H2: dense dataset -> verify -> train -> edit sweep -> eval (fully chained)
df -h "$LORAINV_DATA_ROOT"   # needs ~118 GB free
TRAJ=$(SCRIPT=src/inversion_lora/generate_trajectories_stable_audio.py \
  CONFIG_NAME=generate_trajectories_stable_audio_dense \
  bash $WCSS/submit_trajectories.sh | tail -1 | awk '{print $NF}')
VER=$(TRAJECTORY_DIR=$LORAINV_DATA_ROOT/stable_audio_cosine_ode_fp32_dense991 TOTAL_SAMPLES=1500 \
  bash $WCSS/submit_verify.sh --dependency=afterok:$TRAJ | tail -1 | awk '{print $NF}')
TRAIN=$(SCRIPT=src/inversion_lora/train_stable_audio.py RUN_PREFIX=saocos_ \
  bash $WCSS/submit_train.sh --array=36 --dependency=afterok:$VER | tail -1 | awk '{print $NF}')
EDIT=$(SWEEP_CONFIGS=$WCSS/sao_dense_sweep_configs.sh SCRIPT=edit_stableaudio_medleydb.py \
  LORA_EDITS_SUBDIR=stable_audio \
  bash $WCSS/submit_lora_sweep.sh --array=0-7 --dependency=afterok:$TRAIN | tail -1 | awk '{print $NF}')
ARM=lora SWEEP_CONFIGS=$WCSS/sao_dense_sweep_configs.sh LORA_EDITS_SUBDIR=stable_audio \
  bash $WCSS/submit_eval.sh --array=0-7 --dependency=afterok:$EDIT
```

**Readouts.** H1: `compare_all_methods.py --split genhparam`; the H1 question is whether the LoRA-vs-
no-LoRA delta grows on in-distribution inputs. H2: first the logged dense gap, then val loss vs the
2.630e-4 baseline, then `compare_sao_lora`-style pairing of dense991@3000 against the existing
no-LoRA and saocos_r8@4000 cells. H3: rerun the pooled `compare_all_methods.py`.

---

## 2026-09-14 — RESULT: three nulls, one ceiling. Accuracy is not the binding constraint.

**The headline.** Editing quality saturates by training step 2000. On the 12-checkpoint ladder of
`saocos_r8_a4_lr5e-5` (115 edits, cfg_tar 3.5, 100-step grid): LPAPS 4.4371 at step 2000 vs
4.4370 at step 20000 (tstart 50), 5.2632 vs 5.2630 (tstart 75), against a no-LoRA gap of 0.06 and
0.08. **Ten times more training moves editing by 0.0002** while val loss falls 2.60e-5 -> 2.44e-5.
CLAP, MuQ and directional CLAP equally flat. `output/cycle_and_ladder/`.

So any experiment whose mechanism is "close the shift gap better" is capped before it starts --
which is most of what was queued.

**Three independent nulls, all consistent with that ceiling.**

1. k=1 cycle loss, four gradient-balance targets spanning 20x: matched the *no-cycle baseline* to
   6-7 significant figures (relative spread 3.4e-6 at step 4000). `output/cycle_arms/`.
2. Multi-step cycle k=2 and k=3: 0.001-0.002 LPAPS from baseline at every tstart.
3. 10x more training: 0.0002 LPAPS.

Cause for 1 and 2: `grad L_cycle = (I + (B/A)J^T)(I + (B/A)J)(e-u)` is parallel to `grad L_inv` up
to a `(B/A)||J||` rotation, `B/A = 0.0778` here, and **Adam is per-parameter scale invariant** --
a parallel gradient is a rescaling and gets divided back out. Note k>1 ran with "last step only"
gradients, so its gradient still flows through one prediction; full BPTT is the only untried
variant.

**RETRACTION.** I first reported k=2 improving LPAPS by 0.147. That was my analysis bug: the
regex captured any checkpoint step, so the baseline label pooled step-2000 with step-4000 runs and
pivot_table averaged them into a baseline ~0.15 worse than reality -- manufacturing exactly the
gain. Verified the two supposedly-different runs are identical per example (correlation 1.0, max
diff 0.0). Duplicate (arm, tstart, cfg) cells now assert.

**All methods on one front** (`output/all_methods/`, cfg_tar 3.5, s100). The four ODEInv adapter
variants (LoRA @2000, @4000, cycle k=2, k=3) are one indistinguishable curve spanning 0.002 LPAPS.
DDPM-inv is on a better front: CLAP 0.288 at LPAPS 3.48 where every ODEInv variant needs 4.44 for
CLAP 0.308. SDEdit dominated everywhere.

**Why DDPM-inv wins, from its code** (`tests/test_ddpm_noise_space.py`). `get_zs_from_xts` solves
the sampler's update for the injected noise, so `z` is defined as whatever lands the step on a
prescribed `x_{t-1}`. Tested: exact replay even with the prediction zeroed or 1000x wrong. **It is
not an inversion** -- a free variable per step absorbs all model error, and the code is `x_T` plus
one `z` per step, **101x** the single latent we store. Our matched-NFE table equalises compute and
leaves stored state unequal; report it that way. Neither its exactness nor its edit-friendly noise
statistics is adoptable without becoming DDPM-inversion.

**Running.** cfg35 (experiment 1): 1500 guided trajectories generated and verified, training
capped at 2000 steps (~5.6 h), then an 8-run edit sweep at cfg_src 1.0 *and* 3.5 -- both, because
the adapter trains on guided predictions while the pipeline inverts unguided by default, and
without the pair you cannot separate "better adapter" from "guided inversion". Reconstruction
ladder re-submitted in parallel.

**Owed.** The reconstruction ladder is the missing half of the ceiling argument: it replaces
"editing is flat vs training step" with "flat vs *measured* round-trip error", which is the
difference between a discouraging observation and a publishable null.

**Launch traps that cost real time** (all fixed): `run_lora_sweep.sh` archived into a hardcoded
`audioldm2_ddim`, so every Stable Audio run reported a missing directory after a successful edit
-- `LORA_EDITS_SUBDIR` must reach the *edit* job, not just the eval. Hydra `+key=value` adds a key
while `key=value` overrides one, so a smoke test using `+` passed while the slurm grid could not
work. `SCRIPT` and `CONFIG_NAME` must be set together or the Stable Audio config runs the
AudioLDM2 script. On odeinv, tstart max is 99, not 100.

---

## 2026-09-13 — RESULT: the k=1 cycle loss is inert. Three follow-ups queued.

**Result (closes the cycle-loss question at k=1).** Four arms spanning 20x in gradient-balance
target match the *no-cycle baseline* to 6-7 significant figures at every checkpoint (relative
spread 3.4e-6 at step 4000). Not "the lambda knob does not matter" -- adding the term at all
does not matter. See `output/cycle_arms/`.

Cause, as the algebra predicted: `grad L_cycle = (I + (B/A)J^T)(I + (B/A)J)(e-u)` is parallel to
`grad L_inv` up to a `(B/A)||J||` rotation, and `B/A = 0.0778` on this grid. A parallel gradient
is a rescaling, and **Adam is per-parameter scale invariant**, so it divides the rescaling back
out. That is why setting the gradient norms *equal* still changed nothing. An earlier reading of
mine -- "the Jacobian path dominates, so the arms probe a real axis" -- was wrong: the path is
large in norm and nearly useless in direction.

k=2 and k=3 *do* move the solution (4e-2 relative deviation vs 1e-6). Higher val L_inv, which is
expected since they optimise compounding and val L_inv does not measure it. Editing grid is
`sao_cycle_arms_configs.sh`, all three arms at step 2000 (the deepest depth k=3 can reach).

**Matched-NFE result** (`output/matched_nfe/`). Every method at ~300 denoiser calls, verified by
the in-code counter at 299/300/300. DDPM-inversion dominates the front: CLAP 0.307 at LPAPS 4.17
where ODEInv needs 4.47. The LoRA improves LPAPS at all four depths (-0.024 to -0.080) with CLAP
flat-to-positive -- the hparam-sweep result reproduced on a compute-matched grid -- but that is
~10% of the DDPM gap.

**Why DDPM-inversion wins, established from its code** (`tests/test_ddpm_noise_space.py`).
`get_zs_from_xts` solves the sampler's own update for the injected noise, so `z` is *defined* as
the residual that lands the step on a prescribed `x_{t-1}`. Tested: the replay is exact even with
the prediction set to zero or 1000x wrong. **It is not an inversion** -- reconstruction does not
depend on the model being right, because a free variable per step absorbs the error. ODE
inversion has no such variable: a wrong prediction lands in the latent as exactly `-(B/A)` times
the prediction error.

Consequence for how we report: at T=100 the DDPM code is `x_T` plus one `z` per step, **101x**
the single latent we store. The matched-NFE table equalises compute and leaves stored information
wildly unequal. Neither of its two mechanisms (exactness; the off-trajectory `x_t` that make `z`
image-correlated) is adoptable without becoming DDPM-inversion, so this is a framing change, not
a research direction.

**Queued.**

1. **Is inversion accuracy even the binding constraint?** (`sao_accuracy_configs.sh`, PROBE=recon
   and PROBE=edit.) We close 90.7% of the shift gap and move LPAPS 0.08; that ratio is suspicious.
   The same 12-checkpoint ladder scored for reconstruction and for editing. If editing flattens
   while reconstruction keeps improving, more accuracy cannot help and 2 and 3 below are capped
   before they start. **Run first -- it gates the others.** Note the old `sao_recon_configs.sh`
   is unusable here: it omits LORA_MODE (so it ran the rejected ddim sampler) and names runs by
   checkpoint basename alone, which would have overwritten the sao_r8 ladder step for step.
2. **CFG-aware training** (training config 35, dataset
   `generate_trajectories_stable_audio_cfg35.yaml`). The adapter is a w=1 object deployed at
   cfg_tar 3.5-7.0 against a gap ~3x larger. Both branches are cached so w can be changed without
   regenerating. Costs ~2x generation and 1.5x disk (118 GB) -- **check free space first**.
3. **Second order with a matched adapter.** Staged: first deploy the *existing* adapter with
   order-2 inversion (no training, cheap); only if that moves, train an adapter for the
   second-order substitution. Order 2 is exactly invertible but was 1.29x worse under
   substitution with an adapter trained for the first-order one.

---

## 2026-09-12 — RUNNING: cycle loss (4 adaptive arms + k=2,3) and the matched-NFE comparison

**Hypothesis.** The cycle loss `||z_{i+1} - F_gen(F_inv(z_{i+1}))||^2` is claimed to add
something beyond the inversion loss. Algebraically it does not look like it: both solver steps
are affine, the inverse cancels the generation step's `A`, and the residual collapses to
`B * (D_phi(x_{i+1}) - D_theta(z_hat))`. So at k=1 it is the same two predictions, reweighted.

**Measured first, before spending cluster time.**

- SD1.4 DDIM-50, cfg=1, 150 transitions (`output/cycle_loss_weight/`): the cycle term is
  **0.477% of L_inv** at lambda=1 (1 part in 210). The measured ratio tracks the analytic `B^2`
  across the whole schedule because `||e-v|| / ||e-u|| = 0.985` -- the two terms carry the same
  residual, so the prefactor is the entire story. Weight is a step-size effect: 8.19% at
  DDIM-10, 2.68% at 20, 0.123% at 100.
- Contraction factor `|B/A| * ||J||` = 0.096 << 1, so `L_cycle = 0` has a *unique* zero and it
  is the correct inversion. An earlier degeneracy worry of mine was wrong and is retracted.
- Stable Audio cosine grid (`output/sao_cycle_weight/`): `B^2` is **exactly constant** at
  5.21e-3. The scheduler spaces sigmas geometrically, so every step shares one log-SNR
  increment and `A + B = 1` exactly. On SAO the term carries *no* schedule reweighting at all.

**What that leaves.** The term's only distinct content is the bootstrapped target plus the
gradient path through the frozen teacher. Running jobs measure whether that content is worth
the cost.

**Running (submitted 2026-09-12).**

- Training array 5880219, tasks 29-34. 29-32: k=1 at gradient-balance targets 0.1/0.5/1.0/2.0.
  33-34: k=2 and k=3. All match the baseline `saocos_r8_a4_lr5e-5` (attn, r8/a4, 5e-5) in every
  other field.
- Matched-NFE edits 5880249/5880254/5880264 + dependent evals, BUDGET=300, 16 runs.

**First readings.**

- `lambda_cycle` = 0.0051 at target_ratio 1.0 (0.00051 at 0.1, so the balancer scales
  correctly). That means `||grad L_cycle||` is ~**200x** `||grad L_inv||` before weighting: the
  Jacobian path through the teacher dominates, it is not a small correction. Note this makes the
  naive `lambda ~ 1/B^2 = 192` estimate wrong by ~40,000x -- `B^2` is the *value* ratio, and the
  gradient ratio also carries `(I + (B/A) J)`.
- Step rate is 2.4x my estimate: 10.03 s/step (k=1), 18.77 (k=2), 26.82 (k=3). All six will hit
  the 24 h walltime. Checkpoints land every 2000 steps, so 29-32 reach ~8600 (ckpt 2000-8000),
  33 reaches ~4600 (2000, 4000), 34 reaches ~3200 (**2000 only**).

**Next steps.**

1. Compare every arm at **step 2000**, the point all seven reach (baseline has step_2000 and
   step_4000, both with EMA). Add 4000/6000/8000 where available.
2. Recover k=3 at step 4000 by resuming: task 34 writes
   `training_state_checkpoint_step_2000.pt`, and 2000 more steps is ~15 h, inside one walltime.
3. Matched-NFE figure: the plotter parses the `nfe300_t..._s...` names now
   (`stable_audio_nfe` in `MODELS`, covered by `tests/test_lora_curve_run_names.py`), but the
   x-axis needs choosing -- `steps` is the free variable at a fixed budget, not tstart.

**Traps found on the way** (all fixed, see 845ec32 / 581e1ad):

- `index_for` matched timesteps with `searchsorted`, which needs the query bit-identical to the
  grid after widening. Cached timesteps are float32, the grid float64 -- one rounding from
  silently picking the wrong sigma. Now nearest-neighbour.
- **Gradient checkpointing cannot be used with the cycle loss.** The recompute runs during the
  backward, after the adapter-disabled context has exited, so teacher blocks would be re-run
  with the adapter ON. Fails loudly on a shape mismatch. The option was removed; k >= 2 buys
  memory with batch 4 x accum 8 instead.
- `submit_lora_sweep.sh`'s preview still assumed the old 3-field grid rows, so under `set -u`
  every matched-NFE edit submission died before `sbatch` and the dependent eval then failed with
  "Job dependency problem" on an empty job id.

---

## 2026-08-28 — RESULT: with the objective fixed, the Stable Audio adapter does move editing

First run of the corrected pipeline end to end. Dataset regenerated on Stable Audio's native cosine
sigma grid with the first-order ODE sampler, matched-timestep pairing and data-prediction targets;
adapter `saocos_r8_a4_lr5e-5` (attn, r8, lr 5e-5, batch 8 x accum 4, slurm 5776575); edits through
the new `odeinv` mode on the 115-row hparam split, 24 runs (slurm 5777789), eval 5777813. Report:
`output/sao_lora/20260828_102312/`, figure `output/lora_curves/20260828_102421/`.

Objective: LoRA-disabled val loss 2.630e-4, adapter 2.439e-5 at step 4000 = **90.7% closed**, flat
from step 3000 (2.521e-5) through 17000 (2.334e-5). Step 4000 + its EMA were taken to the sweep.

**Paired against the no-LoRA twin at every cell, same 115 rows:**

| cell | no-LoRA LPAPS | d LPAPS | d mel PSNR | d CLAP |
| --- | --- | --- | --- | --- |
| t25/cfg3.5 | 2.913 | -0.0385*** | -0.089*** | -0.0011 |
| t25/cfg7.0 | 2.995 | -0.0303*** | -0.078*** | -0.0019* |
| t50/cfg3.5 | 4.500 | -0.0624*** | +0.018 | -0.0000 |
| t50/cfg7.0 | 4.925 | -0.0481*** | +0.083*** | +0.0039** |
| t75/cfg3.5 | 5.342 | **-0.0803*** | **+0.140*** | **+0.0050*** |
| t75/cfg7.0 | 5.666 | -0.0660*** | +0.134*** | +0.0017 |
| t99/cfg3.5 | 5.431 | -0.0769*** | +0.129*** | +0.0053*** |
| t99/cfg7.0 | 5.718 | -0.0555*** | +0.141*** | +0.0039** |

Means -0.057 LPAPS, +0.060 mel PSNR, +0.0021 CLAP. EMA matches raw to the third decimal.

The contrast with the beta-grid run is the point. There the largest effect was -0.024 LPAPS **with
CLAP going negative** -- a trade along the front. Here, at six of eight cells, preservation and
alignment improve **together**, so the operating point moves off the front. Sign is consistent at
all eight cells, p<0.001 for LPAPS everywhere. So the earlier Stable Audio null was a property of
the mis-specified objective, not of the model: fix the sampler, the pairing and the target space and
the adapter collects.

### What this does not say

- **Small against the front**, which spans 2.91 to 5.72 LPAPS here. The best cell is 2.9% of it.
  Real, consistent, not a regime change.
- **t25 disagrees with itself**: LPAPS improves while mel PSNR *worsens* (-0.09, p<0.001) at both
  guidances. Unexplained. Shallow inversion is where the adapter has the least to correct, so a
  sign flip there is suspicious and worth a look before the result is leaned on.
- **The odeinv no-LoRA front is worse than the retracted beta front** (5.43 vs 4.28 LPAPS at full
  inversion), consistent with first-order sampling being weaker than the second-order path the old
  numbers used. The adapter is clearing a lower bar. A second-order odeinv with its exact inverse
  would be the fair comparison, and is the obvious next build.
- Checkpoint chosen on **val loss**, which AudioLDM2 showed can peak at a different step than
  reconstruction. No reconstruction eval exists on this path yet.

### Next

1. Second-order odeinv + its exact inverse, so the front is competitive with the old table and the
   `ddpm`/`sdedit` baselines are compared like for like.
2. The reconstruction ladder on this pipeline, which is also the honest checkpoint-selection signal.
3. Real-audio shift gap: everything measured so far is on generated trajectories, while editing
   inverts VAE latents of real mixes.

---

## 2026-08-24 (later) — RETRACTION: the Stable Audio objective is mis-specified

Triggered by listening to the probe wavs: the `beta`-grid teacher sounds clearly worse than the
native cosine one. Chasing that turned up a second, worse defect. Full numbers in
`output/sao_pairing/REPORT.md`, reproduce with `src/inversion_lora/verify_inversion_pairing.py`.

1. **The grids are offset by one step.** The reverse scheduler runs {999, 989, ..., 10}; the inverse
   scheduler runs {0, 10, ..., 989}. Our training pair is `(trajectory[i+1], timesteps[i])`, so at
   100/100 steps the adapter is queried one grid step off what it was trained on.
2. **The target does not control the round trip.** Feeding the inverse solver the exact outputs the
   reverse pass used -- what a perfect adapter predicts -- recovers the initial noise at 0.0153
   relative L2, *worse* than the plain approximation's 0.0080. The two schedulers are not algebraic
   inverses on the same grid, so the shift gap is not the error term. A perfectly trained adapter
   would make inversion ~1.9x worse.
3. **Free 2.2x**: querying the model at the reverse grid's timestep instead drops the error to
   0.0037, with no adapter, one index change in `ddm_inversion/ddim_inversion.py`.

So the section below claiming Stable Audio replicates the AudioLDM2 null result is **withdrawn as
evidence about inversion fidelity**. The measurements stand as measurements -- +1.3 dB
reconstruction, nothing on editing -- but the adapter was not optimising inversion fidelity, so they
say nothing about whether inversion fidelity limits editing on this model. The AudioLDM2 result is
untouched: there one DDIM grid serves both passes and `next_step` is its exact inverse.

Two defects in the same teacher, and they are independent: the schedule mismatch (audible) and the
grid offset (this). Both have to be fixed before any Stable Audio inversion claim means anything.
The principled fix is one piece of work: a deterministic ODE sampler on Stable Audio's *native*
cosine sigma grid plus its exact algebraic inverse, which makes the teacher sound right and the
oracle error go to zero, at which point the existing training pairs are correct by construction.

---

## 2026-08-24 — RESULT: Stable Audio reproduces the AudioLDM2 null result exactly

Both halves measured. Reconstruction ladder: 11 arms x 35 distinct MedleyDB tracks (slurm 5757136
+ 5757198, eval 5757199). Editing grid: 24 runs x 115 hparam rows, 4 tstart x 2 cfg_tar x
{no-LoRA, step 18000, its EMA} (slurm 5757137, eval 5757139). Report:
`output/sao_lora/20260824_103855/`, regenerate with
`python -m editing.compare_sao_lora --root <edits>/medleymd/stable_audio`.

**Reconstruction improves, and saturates by step 2000.** Paired against the frozen teacher:

| arm | d mel PSNR | p | d LPAPS | p |
| --- | --- | --- | --- | --- |
| 2000 | **+1.332** | <0.001 | -0.025 | <0.001 |
| 6000 | +1.312 | <0.001 | -0.016 | 0.004 |
| 10000 | +1.335 | <0.001 | -0.012 | 0.048 |
| 14000 | +1.396 | <0.001 | -0.013 | 0.032 |
| 18000 | +1.378 | <0.001 | -0.016 | 0.010 |
| 18000_ema | +1.370 | <0.001 | -0.014 | 0.021 |

22.62 -> 24.00 dB, and step 2000 is already worth 97% of what step 18000 buys. Compare AudioLDM2's
+1.57 dB. **No dose-response**: the adapter closed 86.3% of the shift gap at step 1000 and 91.9% at
18000, and reconstruction does not distinguish them.

**Editing does not move.** Largest effect anywhere on the grid is -0.024 LPAPS, against a front
spanning 2.79 to 4.93 LPAPS across the eight cells -- about 1% of it. The pattern:

- At tstart=25, small but real preservation gains (LPAPS -0.013/-0.024, mel PSNR +0.20/+0.26,
  p<0.001) **paid for in alignment** (CLAP -0.002/-0.003, p<0.001). A trade along the front, not a
  move off it.
- At tstart>=50 everything collapses into noise or flips sign: LPAPS -0.005..+0.001 (p 0.05..0.85),
  mel PSNR *negative* at every cell, CLAP still slightly negative.
- Raw and EMA are indistinguishable everywhere (differences in the fourth decimal).

Wiring check: the t100/cfg3.5 no-LoRA cell scores LPAPS 4.283 / CLAP 0.287 / mel PSNR 17.380 on
115 rows against the 696-row baseline's 4.326 / 0.286 / 17.048.

### What this settles

The AudioLDM2 finding was not architectural. A second model -- 1.06B DiT, v-prediction, DPMSolver
inversion, a 4.3x rather than ~300x t-dependent shift gap, reaching 92% of it with the plain attn
preset -- reproduces the same three shapes: the objective is learnable, reconstruction improves by
about the same +1.3-1.6 dB, no editing metric moves, and there is no dose-response linking the
first to the second. Inversion fidelity is not what limits editing, on either model.

Also confirmed on the adapter mechanics: injecting a mathematically zero LoRA shifts the DiT output
5.2e-5 relative (cuBLAS rounding from wrapping the Linears), so "LoRA off" is not bit-identical to
"no LoRA" -- 28x below a 2-step adapter's own effect, and far below anything that moves a metric.

### Next

Nothing further on inversion. The untouched lever remains the target-side dynamics -- guidance
schedule, cross-attention control -- and for Stable Audio specifically, the schedule mismatch in
`output/sao_probe/REPORT.md`: its ddim path runs the DiT at timesteps 999..10 where the model was
trained on 0.99..0.19. That affects every SAO inversion baseline in the repo, LoRA or not, and is
now the most suspicious unexamined thing in the Stable Audio results.

---

## 2026-08-23 — RESULT: the objective generalises to Stable Audio Open (91.9%)

The port ran. Dataset: 1500 trajectories, 100 steps, 47.55 s window, `beta` grid (slurm 5746332,
8 shards, **7.6 s per trajectory on an H100**, ~24 min wall, ~79 GB — the 9.5 h estimate was from
an A5000 and was 25x pessimistic). Training: `sao_r8_a4_lr5e-5`, `attn` preset r8 a4 lr 5e-5,
batch 4 x accum 8 (slurm 5746491).

**LoRA-disabled baseline 2.249e-4 over the 30k-transition validation split.**

| step | val/loss | gap closed | val/loss_ema |
| --- | --- | --- | --- |
| 1000 | 3.074e-5 | 86.3% | 8.815e-5 |
| 5000 | 2.219e-5 | 90.1% | 2.232e-5 |
| 9000 | 2.168e-5 | 90.4% | 1.968e-5 |
| 13000 | 1.930e-5 | 91.4% | 1.877e-5 |
| 18000 | 1.832e-5 | **91.9%** | 1.800e-5 |
| 19000 | 1.835e-5 | 91.8% | **1.791e-5 (92.0%)** |

So the shifted-denoiser objective is not an AudioLDM2 artefact: a second architecture (1.06B DiT,
v-prediction, DPMSolver) closes the same share of its shift gap. Two differences worth keeping:

- **SAO reaches 92% with the plain `attn` preset.** AudioLDM2 plateaued at 85-88% across every
  rank and lr and only broke it with `full` + the timestep-embedding modules (92.7%). Whatever made
  the AudioLDM2 gap hard to fit — plausibly its ~300x t-dependence, against SAO's 4.3x — is absent
  here.
- **It converges by ~5000 steps** (90.1%); the remaining 14k bought 1.8 points.

The job was **cancelled at the 24 h walltime at step 19,722/20,000**, not crashed. Step 18000 raw +
EMA + training state are on disk and the last 1000 steps moved val loss 0.03%, so it was not
resubmitted. Throughput note for next time: 4.35 s/step is 8 *sequential* micro-batches of 4;
`batch_size=16 gradient_accumulation_steps=2` keeps the effective batch and should halve wall-clock.

### Where this leaves the question

The objective generalises; **nothing here says editing improves**, and per the AudioLDM2 result
(below) inversion fidelity is not what limits editing on MedleyMD. Before spending anything on the
SAO downstream path, note the probe finding it inherits: the SAO `ddim` edit path runs the DiT on a
linear-beta grid with timesteps 999..10 where the model was trained on 0.99..0.19
(`output/sao_probe/REPORT.md`). Any SAO editing comparison rests on that, LoRA or no LoRA, so it is
the thing to settle first — not another adapter variant.

---

## 2026-08-20 — Port started: Stable Audio Open, objective only

Scope agreed this session: trajectories + training loop + val loss on **Stable Audio Open**. No
reconstruction eval, no editing sweep — the AudioLDM2 result is closed (see below), so the SAO run
only has to show whether the objective is learnable on a second architecture before anything
downstream is built.

### What the probe settled first (`output/sao_probe/REPORT.md`)

- **SAO's native scheduler is an SDE.** `CosineDPMSolverMultistepScheduler` hardcodes
  `sde-dpmsolver++` with an unseeded `BrownianTreeNoiseSampler`: same latent seed, different audio
  (decoded RMS 0.090 vs 0.315). DDIM-style inversion is undefined on it, and diffusers 0.32.2 has
  no deterministic cosine variant.
- **The editing code's `ddim` mode silently changes the noise schedule.**
  `DPMSolverMultistepScheduler.from_config(cosine_config)` drops `sigma_min`/`sigma_max`/
  `sigma_schedule` and falls back to linear betas: sigma 157.4..0.047 instead of 500..0.3, and
  timesteps 999..10 instead of 0.9987..0.1855. The DiT's time embedding takes `log(t)` of a value
  it only ever saw in (0, 1), so every SAO DDIM/inversion baseline queries it ~1000x out of range.
  It still decodes plausible audio (`output/sao_probe/beta_100steps.wav`) — **this is worth a
  listen and probably worth a separate look**; it is not something the LoRA port introduced.
- **Decision: distil the beta-grid DPMSolver teacher**, i.e. exactly what the existing SAO DDIM
  baselines run, so the adapter is comparable with numbers already on disk.
- **The objective has signal there**: shift gap 0.85% relative at t=999 rising to 3.68% at t=10
  (e_RMS 0.0018 -> 0.0163). Flat compared with AudioLDM2, whose MSE varied ~300x across the
  schedule, so `train_max_timestep` is a much weaker lever here.

### What was built

| file | role |
| --- | --- |
| `src/inversion_lora/stable_audio.py` | teacher: conditioning, batched per-example-timestep DiT forward, deterministic reverse trajectories |
| `src/inversion_lora/generate_trajectories_stable_audio.py` | trajectory cache, same on-disk layout as AudioLDM2 |
| `src/inversion_lora/train_stable_audio.py` | `StableAudioInversionTrainer`, a subclass overriding only the forward and the first-batch log |
| `src/inversion_lora/probe_stable_audio.py` | the schedule/shift-gap probe above |
| `config/generate_trajectories_stable_audio.yaml`, `config/train_inversion_lora_stable_audio.yaml` | configs |
| `tests/test_inversion_lora_stable_audio.py` | data-path tests (4) |

`dataset.py` took the only change to shared code: `conditioning_keys` is now a constructor
argument (AudioLDM2's three T5 streams stay the default) plus a `collate_stable_audio_batch` that
needs no padding, because SAO's tokenizer pads to 128 and the two timing embeddings are appended
at generation time, giving a fixed `text_audio` of [130, 768]. `train.py` took a 6-line change:
the first-batch log became an overridable method.

### VERIFIED end to end on an A5000

- 3 trajectories generated at 20 steps, then 6 training steps at batch 4 on the result. Train loss
  falls (0.0116 -> 0.00015 while overfitting 2 trajectories), the LoRA-disabled baseline is
  measured, EMA runs, checkpoints and sidecars are written, `attn` preset injects 384 tensors /
  4.13M parameters over 192 modules.
- Batch 4 at latent [64, 1024] fits in 24 GB at ~1.9 s/step. 78 of 79 tests pass;
  `test_directional.py` fails to import `muq`, which is pre-existing and unrelated.
- Two bugs found by that smoke run, both fixed: the multistep solver carried its step index and
  output history across trajectories (crashed on trajectory 2 and would have corrupted it), and
  the duration conditioning was built with grad enabled, so its graph was freed by the first
  backward.

### Cost

~4.5 s per 20-step trajectory on one A5000, so ~23 s at 100 steps: **1500 trajectories is ~9.5 h
on one A5000**, less on an H100. Storage is fixed by SAO's latent, which is always [64, 1024]
regardless of duration: 53 MB per fp32 trajectory, **~79 GB** for the agreed 1500.

### Next

1. Generate the 1500-trajectory dataset on the cluster (needs a slurm script; the AudioLDM2 one is
   `slurm_scripts/wcss/run_generate_trajectories.sh`).
2. Train `attn` r8 and read the val-loss curve against the LoRA-disabled baseline. If it closes a
   comparable share of the gap to AudioLDM2's 85-88%, the objective generalises.
3. Only then decide whether anything downstream (reconstruction, `attach_inversion_lora` for the
   DiT, editing) is worth building. `apply_lora.py` has **not** been touched for SAO.

---

## 2026-08-15 — RESULT: the inversion LoRA works, and it does not move the editing benchmark

**Both halves are now measured, and they disagree in the most informative way.**

The LoRA does exactly what it was trained to do. At 2500 steps it already improved plain DDIM
reconstruction on held-out audio by **2.8x in latent MSE (5.72e-4 → 2.07e-4) and +9.82 dB mel
PSNR (47.55 → 57.37)**; on real MedleyDB crops, +4.91 dB. Its inverted latents recover the true
initial noise far better than an independent Gaussian draw does (`kl_per_dim` 1.6e-5 against a
0.068 floor).

**On the editing benchmark that improvement is worth nothing.** Taking the fully trained
`r32_a16_lr2e-4/checkpoint_final.pt` (20k steps), applying it to the DDIM inversion pass and
running all 696 MedleyMD edits gives, paired against the like-for-like `cfg_src=1.0` baseline:

| metric | LoRA − baseline | 95% CI | p |
| --- | --- | --- | --- |
| LPAPS | −0.0015 | [−0.0045, +0.0015] | 0.33 |
| mel PSNR | **−0.0485** | [−0.060, −0.037] | <0.001 |
| mel SSIM | −0.0007 | [−0.0013, −0.0002] | 0.013 |
| CLAP | +0.0012 | [+0.0002, +0.0021] | 0.017 |
| MuQ | −0.0008 | [−0.0021, +0.0005] | 0.25 |

The gap to DDPM-inversion is **3.985 dB**. The adapter moved it **−0.049 dB** — 1.2% of the gap,
in the wrong direction, and only detectable at all because n=696.

### Why: inversion fidelity is not the bottleneck

Three independent measurements now say the same thing.

1. **Plain DDIM already reconstructs at 47.6 dB** on the same 200-step grid where its *edits*
   score 14.70 dB. A method cannot be losing 33 dB to inversion error when inversion error is
   that small.
2. **Removing inversion-side guidance entirely changes nothing.** `cfg_src` 3.0 → 1.0 moved
   preservation by −0.068 ± 0.11 dB. That refuted the CFG hypothesis this file recorded on
   2026-08-13.
3. **Making inversion 2.8x more accurate changes nothing**, as above.

The remaining explanation is structural, and it is about how much source information reaches the
reverse pass, not how accurate it is. DDPM-inversion stores **200 per-step noise vectors** and
re-injects source information at every step. DDIM-inversion carries **one latent**, and all of it
must survive 200 steps of `cfg_tar=12.0` guidance toward a different prompt. That is a capacity
difference, which is why improving the accuracy of that single latent does not help.

### The 256-sample sweep: reconstruction improves, noise statistics do not

The learning-rate sweep (job 5696356) ran the larger eval: 256 + 256 fixtures on a 50-step grid,
with the clean latent scored alongside the inverted one as a control.

**Reconstruction improves, and saturates almost immediately.** Against the identical no-LoRA
baseline (generated 8.47e-3 / 35.92 dB, real 1.25e-2 / 34.42 dB):

| lr | gen latent MSE | gen mel PSNR | real mel PSNR |
| --- | --- | --- | --- |
| 3e-4 | 1.53x | +2.16 dB | +1.86 dB |
| 5e-4 | 1.53x | +2.19 dB | +1.85 dB |
| 1e-3 | 1.52x | +2.18 dB | +1.75 dB |
| 2e-3 | 1.51x | +2.16 dB | +1.76 dB |
| 5e-3 | 0.003x | **-17.96 dB** | -16.01 dB |
| 1e-2 | 0.003x | -17.64 dB | -15.78 dB |

Three things follow. The four working rates are indistinguishable, so **learning rate is not
binding** across a 7x range, on top of rank not being binding (r4 matched r32). The ceiling sits
between 2e-3 and 5e-3, where training diverges outright. And it is **converged by step 2500**:
38.04 dB there against 38.12 dB at 20000, so 17500 further steps bought 0.08 dB.

The +2.2 dB here is smaller than the +9.8 dB measured on the 200-step grid because this eval is
out of distribution for the adapter: it learned 5-timestep shifts and is tested on 20-timestep
ones. Both numbers are real; they measure different things.

**`corr_topk` is flat for the entire run.** Generated inverted latents sit at 0.252 before
training and 0.251 after, against a 0.249 iid floor; real audio 0.274 -> 0.273. The single
0.001 move happens by step 2500 and nothing changes over the next 17500. There is no correlation
structure in an inverted latent for the adapter to remove.

The metric is nonetheless working, which the same table demonstrates three ways: the clean-latent
control reads **0.871**, far above the floor; the inverted latents read 0.25, at it; and the
diverging lr=5e-3 run is tracked in real time, 0.251 -> 0.320 -> 0.929 -> 0.991. One small real
signal: real-audio inversions sit ~8x further above the floor than generated ones (0.274 vs
0.252), consistent with real audio gaining less reconstruction (+1.85 vs +2.19 dB).

**The Shapiro-Wilk normality rate was dropped, and an earlier claim retracted.** It assumes iid
samples; latent elements are spatially correlated. Feeding it data that is marginally exactly
N(0,1) but spatially smoothed drives the rejection rate from 4% to 29% purely from correlation,
so it responds to structure rather than non-normality and its p-values are not calibrated here.
At 256 tests the rate also carries a +/-2.7% confidence interval, so the "5.9% -> 5.5%" reported
above is noise, not an improvement. The claim that the inverted latents were "already Gaussian"
rested on that number and is not supported by it; `corr_topk` against its measured floor is the
statement that does hold.

### What this means for the project

- **Do not target "beat DDPM-inv on preservation" with this method.** The measured row is
  `5.995 / 14.587 / 0.349`, on top of the DDIM baseline, and no amount of further training moves
  it: the 20k-step checkpoint performs the same as the untrained one on this benchmark.
- **The reconstruction result stands on its own** and is worth reporting as such: near-exact DDIM
  inversion, +9.8 dB, verified against a no-LoRA baseline measured on identical fixtures.
- If the goal remains the editing benchmark, the lever has to act on the reverse pass — the
  CFG-aware loss variants in [inversion_methods.md](inversion_methods.md), or something that
  carries more than one latent's worth of source information.

---

## 2026-08-13 — Pipeline runs end to end, and the No-CFG loss looks near-degenerate at 200 steps

**Everything is wired and verified. The problem is what it measures.**

Built and tested locally: 24-trajectory dataset (200-step grid, fp32, 1.2 GB), quartile loss
logging, the invert-then-denoise reconstruction eval, W&B online, and WCSS array jobs for both
trajectory generation and a 6-point hyperparameter sweep. First W&B run:
`audio-inversion-lora/runs/d9ogutvi`.

**Eval correctness is verified, not assumed** (`src/inversion_lora/verify_eval.py`): an untrained
LoRA is mathematically the identity (`lora_B = 0`), and it reproduced the no-LoRA baseline on
**29/29 metrics exactly**; perturbing 1024 `lora_B` tensors then moved **17/29**, including
`generated/latent_mse`. So the adapter really does reach the inversion pass, and the eval is
sensitive to it and to nothing else.

### The finding

At the 200-step grid the benchmark uses, **there is almost nothing for the No-CFG shifted-denoiser
loss to learn**:

| quantity | value |
| --- | --- |
| LoRA-disabled loss (the shift gap the LoRA must close) | **4e-6** |
| Plain DDIM reconstruction, generated | **latent MSE 4.2e-4, mel PSNR 48.20 dB** |
| Plain DDIM reconstruction, real audio | latent MSE 6.6e-4, mel PSNR 46.64 dB |

Reconstruction error falls off fast with grid density — 20 steps: 29.1 dB; 50 steps: 36.5 dB;
200 steps: 48.2 dB — because ε(x_{t-1}, t) ≈ ε(x_t, t) becomes an excellent approximation as the
spacing shrinks. At spacing 5/1000 the identity is already nearly optimal, and 20 steps of
training pushed the loss *above* the frozen-teacher baseline (2.7e-5 vs 4e-6).

**So the benchmark's 3.87 dB DDIM-vs-DDPM gap cannot be discretisation error.** Reconstruction at
the same grid is 48 dB while the benchmark's DDIM-inv row scores 14.70 dB mel PSNR. The
difference between the two settings is guidance: the benchmark inverts at `cfg_src=3.0` and
denoises at `cfg_tar=12.0`, and DDIM inversion is known to degrade as inversion-side guidance
grows. The reconstruction eval above uses no CFG.

The quartile breakdown supports this being a real, structured signal rather than noise: the loss
is concentrated almost entirely in the cleanest quarter (q4 4.4e-4 vs q1 ~0), which is where the
shift gap is largest.

### What follows

1. **Diagnostic, cheap, run first:** DDIM inversion on MedleyMD at `cfg_src=1.0`, everything else
   identical to the existing DDIM-inv row (baselines array index 6, eval index 6). If preservation
   jumps toward DDPM-inv, the gap is confirmed as CFG-driven and the No-CFG loss is targeting the
   wrong error.
2. **Reconsider the loss.** [inversion_methods.md](inversion_methods.md) has six variants; only
   No-CFG is implemented. A CFG-aware variant is what the benchmark gap actually calls for.
3. The trajectory dataset and sweep are still worth generating — they are reusable across loss
   variants, since the cached targets are conditional epsilon on the same grid. Only
   `save_uncond_target: true` is needed for the CFG variants, and that costs a second teacher
   forward per step, so decide before generating the full corpus.

---

## 2026-08-12 — Benchmark reproduction DONE; LoRA path is next

The goal of this phase was reproducing the existing benchmark, and that is finished. Six
baselines (DDPM-inv / DDIM-inv / SDEdit × AudioLDM2 + Stable Audio), 696 edits each, scored and
written up in [medleymd_baselines.md](medleymd_baselines.md).

**The measured target:** on AudioLDM2, DDIM-inversion trails DDPM-inversion by **3.87 dB mel
PSNR** (14.70 vs 18.57) and 25% on LPAPS. On Stable Audio, **4.43 dB** (17.05 vs 21.48). That is
the headroom the inversion LoRA has to recover — and it must do so without losing adherence,
since AudioLDM2 DDIM currently has the *best* MuLan of its three methods.

Decisions taken to close this phase:
- **FAD skipped** on this benchmark; reported as `nan`. Ill-conditioned because MedleyMD has only
  35 unique excerpts. Not chased further.
- **tstart sweep deferred.** The partial-`tstart` DDIM variants from `stable_audio_edits.sh` are
  not run; revisit on a validation split or another benchmark later.
- **Full 696 rows**, not the `test` split. See medleymd_baselines.md for why that is safe here
  and where it stops being safe once the LoRA introduces hyperparameters.

### Next, in order

1. `reconstruct.py` — LoRA-DDIM vs plain DDIM invert→denoise on real MedleyDB audio, reporting
   mel PSNR/SSIM and latent L2. **This is the go/no-go.** GPU-free to write. Measure on both
   held-out synthetic trajectories and real audio: the gap between them is the synthetic/real
   distribution shift, and tells us whether real-audio-seeded trajectories are needed.
2. Generate ~1.5k MusicCaps trajectories (`generate_trajectories_gonogo.yaml`). ~300k teacher
   forwards; hours on an H100 at the rates measured for the baselines, not the days estimated
   from the A5000.
3. Train (`train_inversion_lora.yaml`), then run the go/no-go and record the numbers here
   **before** touching editing metrics.
4. Only if it passes: edit with the LoRA and score it as a seventh row in the AudioLDM2 table.

---

## 2026-08-11 — Baselines moved to WCSS

Started the six baselines locally, then **cancelled after 7.7 h** (493/4176 edits) to re-run on
WCSS. Local partial outputs carry an `INCOMPLETE_PARTIAL_RUN.txt` marker and are **not** results.

**Submit on WCSS (PWR `lem`)** — array job, 6 configs × 12 shards = 72 tasks:

```bash
cd /lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/code/lorainv/audio
module load Python/3.10.4-GCCcore-11.3.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_lorainv.txt
# .env: MEDLEYDB_AUDIO_DIR, HF_TOKEN, HF_HOME, EDIT_OUTPUTS_DIR  (see .env.example)
python editing/AudioEditingCode/code/slurm_scripts/wcss/prefetch_models.py   # login node
mkdir -p outputs/logs/slurm
sbatch --account=$HPC_PWR_ACCOUNT --partition=$HPC_PWR_PARTITION \
  editing/AudioEditingCode/code/slurm_scripts/wcss/run_baselines_medleymd.sh
```

Cluster facts that shaped the script: `gpu:hopper:1` (H100, so the A5000-measured 280 s/edit is
pessimistic by ~2-3×), `Python/3.10.4-GCCcore-11.3.0` (all our code verified 3.10-compatible;
the pins are version-based so cp310 wheels resolve fine), venv not conda, `lem-gpu-short` with
5 h walltime, and a retry loop because transient failures are common there.

`HF_HUB_OFFLINE=1` is exported in the job on the assumption that compute nodes have no
internet. `load_model` falls back from `local_files_only=True` to `False`, so without a warm
cache all 72 tasks would each hang on a network timeout rather than fail fast. `prefetch_models.py`
warms AudioLDM2-large (~7 GB) and Stable Audio Open (~5 GB) on the login node first. Unset
`HF_HUB_OFFLINE` if the nodes are actually online.

Keep `HF_HOME` and `EDIT_OUTPUTS_DIR` on lustre — `PATH_EDIT_OUTPUTS` defaults to `audio/outputs`
inside the repo, and 4176 WAVs on a quota-limited `$HOME` would be a problem.

**Configs** (both models × DDPM-inv / DDIM-inv / SDEdit, full 696 rows):

| Model | steps | cfg src/tar | tstart |
| --- | --- | --- | --- |
| AudioLDM2-large | 200 | 3.0 / 12.0 | 100, DDIM 200 |
| Stable Audio Open | 100 | 1.0 / 3.5 | 50, DDIM 100 |

DDIM uses `tstart == steps`: partial inversion is not DDIM inversion, and this is the exact
baseline the LoRA is meant to improve.

**Measured throughput** (use these for walltime, not guesses): AudioLDM2 **~280 s/edit**,
Stable Audio **~94 s/edit**. My earlier 162 s/edit estimate was wrong by 1.7× because it scaled
the smoke edit by audio duration, but per-edit fixed costs do not shrink with duration. Full run
≈ 149 GPU-hours.

**Two bugs fixed — `edit_stableaudio_medleydb.py` had never been runnable:**
1. Mixed import roots: `from stable_audio_run import` needs `code/` on `sys.path`,
   `from editing.AudioEditingCode.code.env import` needs `audio/`. It failed from either cwd.
2. `"-".join([str(x) for x in cfg_src])` on a `float` → `TypeError`, fired with default args
   because `run_name` defaults to `None`.

**Known waste, worth fixing before the LoRA runs:** `load_model` sits *inside*
`run_audioldm_edit`, so the model is reloaded from disk for every edit — roughly 20 GPU-hours
(~13%) across a full 4176-edit sweep. Also a hardcoded `time.sleep(5)` per edit.

**Also note:** Stable Audio writes to `outputs/edits/medleymd/**medleymd**/stable_audio/...` —
the driver appends `dataset_name` to a path already containing it, so the two models' output
layouts differ. Left as-is rather than silently changing the layout; the eval step must handle it.

---

## 2026-08-10 — Data + dataset landed, blocked on GPUs

**Decision: go/no-go first.** Train a rank-8 LoRA on ~1.5k MusicCaps trajectories and measure
**reconstruction error on real MedleyDB audio** (LoRA-DDIM invert → denoise vs plain DDIM)
before spending anything on the full corpus or on editing metrics. If LoRA inversion does not
beat plain DDIM inversion on real audio, corpus size will not rescue it.

### Done

| Piece | Where | State |
| --- | --- | --- |
| Env (torch 2.4.1 / diffusers 0.32.2 / transformers 4.47 / peft 0.15.2) | `audio/.venv`, pins in `audio/requirements_lorainv.txt` | works |
| Trajectory generation + target caching | `audio/src/inversion_lora/generate_trajectories.py` | smoke-tested on CPU |
| Dataset completeness verifier | `audio/src/inversion_lora/verify_trajectories.py` | exits 1 on partial samples |
| Dataset + T5-padding collate + trajectory-level split | `audio/src/inversion_lora/dataset.py` | 11 unit tests pass |
| Trainer (no-CFG loss, ckpt/resume, W&B) | `audio/src/inversion_lora/train.py` | overfits smoke set on CPU |
| Go/no-go run config | `audio/config/generate_trajectories_gonogo.yaml` | ready to fire |
| Train config | `audio/config/train_inversion_lora.yaml` | ready |

### Verified, not assumed

- **DDIM invariant exact**: stepping `trajectory[i]` with `target_eps[i]` reproduces
  `trajectory[i+1]` to `0.0e+00`. The cache really is a DDIM trajectory.
- **Cached target is the teacher ε at the noisier latent**: `0.0e+00`.
- **The gap the LoRA must close** (teacher ε at the *cleaner* latent vs cached target):
  `9.3e-02` max-abs. Non-degenerate — use this to sanity-check training loss scale.
- **Disabling LoRA adapters restores the frozen teacher bit-exactly** (max diff `0.0`, 1024
  injected modules). Required for cached-target training and for
  invert-with-LoRA / generate-without.
- **Batched mixed-length T5 conditioning == per-item forwards** (rel `2.6e-07`), so the
  padding collate is safe.
- AudioLDM2-large: ε-prediction, `DDIMScheduler`, 200 steps (996→1, spacing 5), 718M-param
  UNet, latents `[8, 256, 16]` at 10.24 s → **50 MiB/sample fp32**. rank-8 LoRA on
  `to_q/to_k/to_v/to_out.0` = 7.68M trainable params (2048 tensors).
- **Gradients flow through the vendored `unet_forward`** into LoRA params, with **zero leakage
  into frozen weights**. At init only `lora_B` has gradient (PEFT zero-inits B, so A's grad is
  0 on step 0) — expected, not a bug.
- **The LoRA learns**: on the 16-transition smoke set, loss goes from the LoRA-disabled
  baseline **0.1180** to **4.7e-04** (~250×), so both A and B train. Baseline MSE 0.1180 is the
  number to beat; anything at or above it means the adapter is not helping. The trainer logs it
  at startup on every run.
- CPU throughput ~15 s/step at batch 2 (context for why GPUs gate everything).

### Blocked / missing

1. **GPUs** — 8× A5000 but only ~10 GB free per card (another job holds 13.7 GB). The go/no-go
   generation is 1500 × 200 = **300k teacher forwards**; infeasible on CPU. Nothing on the
   critical path can start until cards free.
2. **Real audio** — expected 2026-08-10 (+1h from writing). Two separate needs, do not conflate:
   - **MusicCaps audio** → lets trajectories start from real audio instead of random noise
     (see "Real-audio trajectories" below). Optional for the go/no-go.
   - **MedleyDB V1** → *required* for the go/no-go, since the whole question is reconstruction
     error on real music. Licensed request.
3. ~~`env.py` placeholders~~ **resolved**. Repo-relative paths derive from `__file__` (so they
   survive a `git pull` onto another server); dataset dirs come from `audio/.env` overrides.
   Verified all **696/696** benchmark rows resolve to an existing mix file.

   | Path | Value |
   | --- | --- |
   | `PATH_AUDIOS_MEDLEY` | `/nas/lstanisz/data/medleydb/V1_mix` (`MEDLEYDB_AUDIO_DIR`) |
   | `PATH_PROMPTS_MEDLEY` | `MedleyMDPrompts/captions_gpt5.csv` — the **only** CSV with all four driver columns; `captions_targets_with_sources.csv` has no `edit` column and would `KeyError` on the hook lookup |
   | `PATH_EDIT_OUTPUTS` | `audio/outputs/edits` |
   | `PATH_LOWER_BOUND_MEDLEY` | `audio/outputs/medleymd/lower_bound/audios` (still to be generated) |
   | `ALDM2_TEMP_DIR` | `audio/.temp/audioldm2` |
   | `PATH_MUSICCAPS` | `audio/data/musiccaps/audio` (not populated) |

   `audio/.gitignore` already covers `outputs/`, `data/*`, `.temp`. All values stay `str`
   because callers do `PATH_EDIT_OUTPUTS + "/medleymd"`.
4. ~~Trainer~~ **done** — verified by overfitting the smoke set on CPU.
5. **Reconstruction eval not written** — this *is* the go/no-go measurement. Next code task.
6. **Metric checkpoints** for the editing comparison (step 5, not the go/no-go):
   `music_audioset_epoch_15_esc_90.14.pt` under `res/clap/pretrained`, `OpenMuQ/MuQ-MuLan-large`.
7. **FAD/metrics env** not built (`requirements_fad_*.txt`: `audioldm_eval`, `ssr_eval`, skimage).

### Decisions taken (and why)

- **All AudioLDM2 work in the audio env, not the root env.** The root env (torch 2.11 /
  diffusers 0.37 / transformers 5.6) cannot run AudioLDM2 at all: `get_text_features` returns
  a `ModelOutput`, `generate_language_model` is dead (`GPT2Model` has no
  `_update_model_kwargs_for_generation`), and `encode_prompt` hits a float/double mismatch.
  Independently correct anyway — same env *and* code path as the baselines keeps the frozen
  teacher bit-identical, which the comparison depends on.
- **Cost: the trainer duplicates ~200 lines of `SDXLInversionTrainer`** rather than importing
  it. Forced by the env split, not a design choice.
- **No-CFG loss only** for now (1 of the 6 in `inversion_methods.md`). Trajectories are
  generated at `guidance_scale=1.0`, so trajectory and loss are self-consistent.
- **`save_uncond_target=false` for the go/no-go** (it doubles generation cost and no-CFG does
  not use it); **true for the full corpus**, so the CFG variants never need a regeneration.
  Note the SDXL pipeline never cached uncond targets, which is exactly why the CFG losses in
  `inversion_methods.md` were never runnable there.
- **`device` is explicit config, default `cpu`.** The original auto-select grabbed a shared GPU.
- **200 steps** for training data, matching `edit_audioldm_medleydb.py`, so the LoRA is trained
  on the same DDIM grid the baselines are evaluated on.
- **`meta.json` is the completion sentinel**, written last via atomic rename. An earlier CUDA
  OOM died in the vocoder *after* `target_eps.pt` was written and resume then silently skipped
  the half-written sample while reporting success. At 5.5k samples with preemption this would
  have quietly poisoned the dataset.

### Landmines to remember

- **`src/metrics/alignment.py:160`** — on filename mismatch `calculate_psnr_ssim` returns
  `{"psnr": -1, "ssim": -1}` instead of failing. A misnamed reference dir yields `-1` in the
  results JSON, not an error. Wrap with a strict filename-overlap pre-check.
- **`src/metrics/alignment.py:344`** — bare `except Exception: continue` drops files from the
  FAD/KL feature set with no count. FAD could silently be computed over a subset. Assert the
  processed count equals the file count.
- **`PATH_LOWER_BOUND_MEDLEY` must be per-example wavs named `a{idx}.wav`** in
  `prepare_dataset` row order — PSNR/SSIM are *paired* and gated on >99% filename overlap via
  `get_filename_intersection_ratio`. Open question: reference = raw source audio, or the VAE
  round-trip of the source? Round-trip is the *achievable* ceiling and isolates edit drift;
  raw source conflates codec loss with edit drift. Plan: generate both, report vs round-trip.
- **The lower-bound wavs must be truncated exactly like the edits.** MedleyDB mixes run
  **13.1 s – 302.8 s** (mean 60.8 s, 44.1 kHz stereo, 375 MB), and **14/35 exceed 60 s** so
  `audioldm_run.create_truncated_audio()` cuts them to 60 s. A reference built from the *full*
  source would make mel PSNR/SSIM compare e.g. 60 s against 303 s. `eval_medley.py` has its
  sample-rate/length assertion **commented out** (:36-40) and LPAPS/CLAP window internally, so
  this mismatch would not raise — it would just silently produce wrong paired numbers.
- **Benchmark class balance is extremely skewed**: GENRE 397, INSTR 195, MOOD 34, VOICE 31,
  OTHER 27, **TEMPO 12** (696 total over 35 tracks). GENRE+INSTR is 85%, so any headline
  average is effectively a GENRE score and the four small classes are directional only.
- **DDPM-inversion cannot be beaten on fidelity** — `inversion_forward_process` does not
  invert: `sample_xts_from_x0` draws xts straight from x0 and `get_zs_from_xts` solves for the
  noise making the reverse step exact. Reconstruction is exact *by construction*. So the honest
  framing is **LoRA-DDIM vs plain DDIM inversion, with DDPM-inversion as the fidelity
  reference**, not "we beat everything".
- **`models.py:1518-1541`** — `load_model` has no `else`; an unknown `model_id` gives
  `NameError` on `ldm_stable`, and it calls `edit_method.lower()` despite the `None` default.
- **T5 conditioning length is prompt-dependent** (46 vs 128 observed); GPT-2 stream is fixed
  at `(8, 768)`. Long MusicCaps captions get truncated at 128 tokens — same as the baselines.
- Dead code in `diff_inversion/modeling/train.py`: `_teacher_mode` (:439) and `encode_prompts`
  (:245). `_teacher_mode` is the hook to use if we ever switch to online teacher targets.

### Real-audio trajectories (once MusicCaps audio lands)

The shifted-denoiser loss needs *exact* DDIM pairs, so do **not** train on a raw DDIM-inversion
trajectory of real audio — those pairs only hold approximately and would inject the very error
we are removing. Correct construction: real x0 → DDIM-invert with the teacher to x_T → teacher
**denoise** from x_T saving that trajectory. Pairs stay exact by construction and x0' sits near
real audio, narrowing the synthetic/real gap. Costs 2× forwards per sample. ~10 lines in the
generator (pluggable initial latent), not a redesign.

### Next actions, in order

1. `reconstruct.py` — LoRA-DDIM vs plain DDIM invert→denoise on real audio; report mel-domain
   PSNR/SSIM + latent L2 vs the source. **This is the go/no-go.** GPU-free to write.
2. When GPUs free: shard the 1.5k generation, then `verify_trajectories.py --check_step 16`
   as a hard gate before training. Local runs must be detached:
   `setsid bash -c 'CUDA_VISIBLE_DEVICES=N ... > run.log 2>&1' </dev/null &`
3. Train (`train_inversion_lora.yaml`, ~20k steps), then run the go/no-go and record the
   numbers **here** before touching editing metrics.
4. Only if the go/no-go passes: full corpus with `save_uncond_target=true`, then the editing
   comparison and the metrics/FAD envs.

### Deliberately skipped (revisit if needed)

- **Final-tail timestep oversampling** (`sampling=final_tail` on the SDXL side, with its own
  test). Plausibly matters for audio too, since inversion error concentrates at low noise, but
  not needed to answer the go/no-go. Uniform sampling for now.
- **LR scheduler** — constant LR. The SDXL side has cosine/constant configs.
- **`active_steps` / `active_fraction`** (LoRA on only the first K inversion steps). Already
  proven useful on the image side and likely important for audio given the synthetic/real gap.
  Belongs in the reconstruction eval as a swept knob, not in training.
