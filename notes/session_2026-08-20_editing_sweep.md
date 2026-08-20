# Session notes — hparam sweep, LoRA evaluation, and output layout

**Date:** 2026-08-17 → 2026-08-20 (one session, continues earlier work in the same conversation)
**Conversation ID:** `cb1a256d-aed0-4ac2-b386-49c65bbbb606`
**Session:** https://claude.ai/code/session_01VtLGeWNuakTRHKN5PtpRE9
**Commits:** `b846eb5` … `d2ec338` on `audio_edit`

What happened, in order: trained the inversion LoRA on the cleanest quarter of the schedule only;
built and ran a 48-run three-method hyperparameter sweep on the 115-row MedleyMD split; ran six
LoRA checkpoints over the DDIM part of that grid; and reworked the output layout after the edit
outputs turned out to be inode-bound. The scientific conclusions are in
[audio_inversion_lora_status.md](audio_inversion_lora_status.md); this file records the decisions,
the details that are easy to get wrong, and every bug found along the way.

---

## 1. Q4-only training (`train_max_timestep`, `num_loss_bands`)

**Ask:** train on the cleanest 25% of the schedule only, with the loss split into 5% bands.

**Decision — one mechanism, two scalars.** Rather than a configurable list of band edges,
`num_loss_bands` splits `[0, train_max_timestep]` into equal bands named by their share of the
full schedule. The default (`null`, 4) reproduces the old quartiles exactly; `250, 5` gives
`q25_20 … q05_00`. This avoids passing a bracketed list through sbatch (glob/quoting hazard) and
keeps one code path. Boundaries are half-open on the noisy side, matching the previous
convention, so `t/T = 0.25` belongs to the cleanest band. Verified: the 200-step grid lands
50/50/50/50 by quarter and 10 per band within the last quarter.

**Decision — validation stays on the full schedule.** A restricted run has to show what it does
to the 75% it no longer sees, and full-schedule `val/loss` stays comparable with the 24 earlier
runs. Costs little: the measured band losses on a full-schedule run were 0.000000 / 0.000004 /
0.000016 / 0.000173 for q100_75 … q25_00, so the cleanest quarter already carries ~90% of the
loss mass and full-schedule `val/loss` is dominated by it.

**Side effect:** W&B keys for unrestricted runs changed from `train/loss_q1..q4` to
`train/loss_q100_75..q25_00`. Old panels will not line up.

**Result — it overfits.** 71k transitions instead of 285k means 9 epochs instead of 2.2. Real-audio
reconstruction peaks around step 7500 (0.00884 latent MSE, 35.99 dB), is level with the no-LoRA
baseline (0.01247) by 12500 (0.01198), and is worse than it by 20000 (0.01389) — while training
and validation loss keep falling the whole time. **Pick checkpoints on
reconstruction, never on loss.** Saves are every 2000 steps, evals every 2500, so the peak is not
a saved checkpoint — 6000 and 8000 bracket it.

## 2. The hyperparameter sweep

**Design.** `tstart` is the shared strength knob because it means the same thing in all three
methods — the fraction of the trajectory regenerated under the new caption:

| method | inversion steps | reverse steps |
| --- | --- | --- |
| DDPM-inv | 200 (always) | `tstart` |
| DDIM-inv | `tstart` | `tstart` |
| SDEdit | — | `tstart` |

Grid: 3 methods × `tstart` {50,100,150,200} × `cfg_tar` {3,6,12,18} = 48 runs × 115 edits.
`cfg_src` fixed at 3.0 — 3.0 vs 1.0 moved DDIM preservation by 0.068 dB, so it does not earn a
dimension. The three published operating points are grid members, which is what bridges the sweep
to the 696-row table.

**Decision — one task per config, no sharding.** The slowest config is ~4.1 h against an 8 h
limit, and `SKIP_EXISTING=1` makes a killed task resumable. Sharding was offered for wall clock
and not taken.

**Cross-check that mattered:** the 115-row split reproduces the full benchmark almost exactly
(DDIM t200/cfg12 LPAPS 6.071 here vs 6.070 on 696 rows; DDPM t100/cfg12 4.847 vs 4.841), so the
subset is representative and the sweep's conclusions carry.

## 3. The `--split` plumbing

`env.py` calls `load_dotenv(override=True)` — deliberately, so a stale exported variable cannot
beat `.env`. Consequence: **you cannot point a job at a different prompt CSV with an environment
variable.** Hence an explicit `--split` flag on the drivers and the eval, with `env.py` owning the
split table and resolving a split to `(prompts CSV, paired reference)` so the two can never
disagree. The eval now derives the reference directory *and* the expected row count from the
split; the hardcoded 696/35 are gone.

Grid definitions live in one sourced `.sh` per sweep, read by both the edit job and the eval job,
so run directories are derived rather than retyped. This follows the earlier `lora_run_name.sh`
incident: SLURM spools only the batch script, so helpers must be sourced from `$SLURM_SUBMIT_DIR`,
never via `BASH_SOURCE`.

## 4. Bugs found

1. **Subset references were misnamed.** `build_lower_bound` named reference wavs by *position*
   (0..114) while the drivers name outputs by the row's *original index* (`a178.wav`). Only the
   full split coincides, so `hparam`/`test`/`loc` references had zero filename overlap and
   `calculate_psnr_ssim` returns −1 without raising. Fixed to always use `df.index`; a test
   asserts the reference name set equals the driver's output name set. Nothing already scored was
   affected — only `full` and `tracks` had ever been built.
2. **The atomic wav write was broken off-cluster.** The temp name `a178.wav.partial.<pid>` has no
   `.wav` suffix, and torchaudio's ffmpeg backend picks its muxer from the extension, so every
   edit died at save. WCSS's soundfile backend accepted it, which is why the cluster runs worked.
   The temp now keeps `.wav` **and** lives in a `.partial/` subdirectory, because the paired
   metrics glob every `*.wav` next to the edits and would score a leftover.
3. **`time.sleep(5)` per edit** in `create_truncated_audio` — 8% of a DDIM edit, ~10 GPU-h across
   the sweep. The file is written by a synchronous local `torchaudio.save` and read back by the
   same process, so POSIX read-after-write already covers it. Replaced with a size assert.
4. **`load_model` runs per edit.** `run_audioldm_edit` loads the 12 GB pipeline itself and the
   driver calls it once per row. Only ~2 s once the page cache is warm, so it was left alone —
   hoisting it would change the RNG stream and DDPM's inversion draws noise.
5. **CFG always runs two forwards.** `get_noise_pred` computes the unconditional prediction
   unconditionally, then blends it with weight 1 at `cfg_scale=1.0`. The 35-track reconstruction
   runs paid exactly 2× for nothing.
6. **`set_reproducability` is order-dependent.** It sets `matmul.allow_tf32 = False` and then
   `set_float32_matmul_precision("high")` two lines later, which turns matmul TF32 back on.
   `cudnn.allow_tf32` stays off. Net effect versus PyTorch defaults: convolutions lose TF32.
   Worth only 3%, so not changed.
7. **PEFT's `disable_adapters` takes precedence over `merged` and silently unmerges.** A
   merged-but-disabled layer loses the adapter on its next forward. `attach_inversion_lora`
   therefore keeps adapters *enabled* while merged and lets `merged` alone gate the branch;
   `tests/test_lora_merge_toggle.py` pins this so a PEFT upgrade cannot break it quietly.
8. **Unfused `full`-preset adapters cost 2.3× per edit** (286.6 vs 124.3 s on the cluster) because
   the adapter is a side branch on 1467 modules. Merging removes the compute entirely: 660 ms vs
   651 ms for no adapter at all, at an epsilon difference (9.1e-5) no larger than the 5.5e-5 that
   routing through a *disabled* adapter already costs under TF32.
9. **My own misread of the timing logs.** `baselines-5682067_48` is Stable Audio, not AudioLDM2 —
   the array packs 12 shards per config, so task N is config N/12. This understated the sweep by
   30% (85 → 115 GPU-h). Always divide by `N_PARTS` before reading a config off an array index.
10. **`psnr_ssim_per_file.csv` row order follows the directory listing**, so it differs between an
    archived and a non-archived scoring of the same run. Every reader here joins on filename;
    anything comparing positionally would silently compare different examples.

## 5. Output layout (inodes)

`outputs/edits` held **36,685 files** across 109 runs: 18,172 edits, 17,514 in 32 kHz resample
caches, 999 metrics. Three changes take that to ~440:

- **One `audios.tar` per run** (116 inodes → 2). `archive_run.py` verifies every file is present at
  the right size before removing anything; a test drops a member mid-write and asserts the
  directory survives.
- **The eval unpacks to node-local scratch.** The metrics need a directory (`audioldm_eval`'s
  PSNR/SSIM and FAD passes are vendored and take paths), and because `ensure_resampled` writes its
  cache as a *sibling* of the audio directory, the 32 kHz copies land on scratch and never reach
  lustre. The CSVs are copied back afterwards.
- **Metrics 8 files → 3.** The four per-example CSVs share one key (position in the split) so they
  merge; the three aggregate JSONs merge. `psnr_ssim_per_file.csv` stays separate because it is
  keyed by wav filename, i.e. the row's index in the full set — joining the two needs the split
  CSV to map one to the other. `run_metrics.per_example` names the index it returns (`position` vs
  `row_idx`) so the two cannot be silently joined.

**Rejected:** packing the `lower_bound_*` references. They are shared across array tasks, so
packing them would make every task unpack and re-resample its own copy instead of sharing one
cache — 1,692 inodes is not worth that.

**Verification:** all 109 runs' metric files were mirrored locally, migrated, and both figures
regenerated — the sweep table diffs clean against the pre-migration version and the LoRA paired
deltas are identical.

## 6. Constraints recorded

- **The 60 s input window is fixed.** It is 94% of the cost (the UNet runs on a latent 5.9× the
  model's native 10.24 s window) and truncating it was ruled out on 2026-08-17. Do not re-propose
  it as a speedup.
- `*-mount/` is read-only: test locally, commit, push, and the user pulls on the cluster.
- Everything on WCSS goes through `sbatch`; there is no GPU or Python environment on the login
  node.

## 7. Open items

- Regenerate `output/tables/20260817_medleymd_editing.tex` with matched-`tstart` rows. Its caption
  still asserts the 3.94 dB gap that the matched-`tstart` measurement in section 2 retired.
- **Reconstruction floor on the `hparam` split** (~4 GPU-h): `--reconstruct --cfg_src 1 --cfg_tar 1
  --tstart 200`. DDIM reconstruction of real audio scores LPAPS 3.881 on the 35-track split and
  the best-preserving *edit* scores 3.883 on the hparam split — different splits, so possibly
  coincidence, but if it holds then the left edge of the entire front is the VAE + vocoder
  round-trip floor and no inversion of any quality can go left of it.
- fp16 autocast: 3.95× on an A5000 at 1.0e-3 relative epsilon error. Needs the H100 ratio
  (`python -m editing.profile_unet_forward`) and a 20-row fp32-vs-fp16 A/B on the metrics before
  it can be trusted for a sweep.
- Inversion caching: the inversion depends only on (audio, source caption), which repeat 6.5× in
  the 696-row split. Exact for DDIM (no RNG consumed), ~1.8× off a DDIM run.
- Migrate the 109 scored runs and drop the caches (commands in the session; both are destructive
  and were left to the user).
