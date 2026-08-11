# Audio Inversion LoRA — Status & Open Items

Porting the shifted-denoiser inversion LoRA (see [inversion_methods.md](inversion_methods.md))
from SDXL/SD1.5 to **AudioLDM2**, to be compared against the DDPM-inversion / DDIM / SDEdit
baselines in `audio/editing/` on the MedleyMD editing benchmark.

Most recent first. Keep this file current — it is the handover doc between sessions.

---

## 2026-08-11 — Baselines moved to WCSS

Started the six baselines locally, then **cancelled after 7.7 h** (493/4176 edits) to re-run on
WCSS. Local partial outputs carry an `INCOMPLETE_PARTIAL_RUN.txt` marker and are **not** results.

**Submit on WCSS:** `audio/editing/AudioEditingCode/code/slurm_scripts/wcss/run_baselines_medleymd.sh`
— 6 configs × 12 shards = 72 jobs. Dry-run with `SUBMIT=0`. Needs `GRANT_ACCOUNT` and
`GRANT_PARTITION`, plus a cluster-side `audio/.env` holding `MEDLEYDB_AUDIO_DIR` and `HF_TOKEN`
(Stable Audio Open is license-gated), and a venv from `audio/requirements_lorainv.txt`.

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
