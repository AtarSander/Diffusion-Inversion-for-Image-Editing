# Audio Inversion LoRA — Status & Open Items

Porting the shifted-denoiser inversion LoRA (see [inversion_methods.md](inversion_methods.md))
from SDXL/SD1.5 to **AudioLDM2**, to be compared against the DDPM-inversion / DDIM / SDEdit
baselines in `audio/editing/` on the MedleyMD editing benchmark.

Most recent first. Keep this file current — it is the handover doc between sessions.

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
| Dataset + T5-padding collate | `audio/src/inversion_lora/dataset.py` | 7 unit tests pass |
| Go/no-go run config | `audio/config/generate_trajectories_gonogo.yaml` | ready to fire |

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
  `to_q/to_k/to_v/to_out.0` = 7.68M trainable params.

### Blocked / missing

1. **GPUs** — 8× A5000 but only ~10 GB free per card (another job holds 13.7 GB). The go/no-go
   generation is 1500 × 200 = **300k teacher forwards**; infeasible on CPU. Nothing on the
   critical path can start until cards free.
2. **Real audio** — expected 2026-08-10 (+1h from writing). Two separate needs, do not conflate:
   - **MusicCaps audio** → lets trajectories start from real audio instead of random noise
     (see "Real-audio trajectories" below). Optional for the go/no-go.
   - **MedleyDB V1** → *required* for the go/no-go, since the whole question is reconstruction
     error on real music. Licensed request.
3. **`audio/editing/AudioEditingCode/code/env.py` is all placeholders** — six paths, none set:
   `PATH_AUDIOS_MEDLEY`, `PATH_PROMPTS_MEDLEY`, `PATH_LOWER_BOUND_MEDLEY`, `PATH_MUSICCAPS`,
   `ALDM2_TEMP_DIR`, `PATH_EDIT_OUTPUTS`.
4. **Trainer not written** — next code task. GPU-free to write; verify by overfitting the
   16-transition smoke set on CPU.
5. **Reconstruction eval not written** — this *is* the go/no-go measurement.
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

1. `trainer.py` — no-CFG loss, LoRA r=8, ckpt/resume, W&B. Verify: overfits the 16-transition
   smoke set to ~0 on CPU.
2. `reconstruct.py` — LoRA-DDIM vs plain DDIM invert→denoise on real audio; report mel-domain
   PSNR/SSIM + latent L2 vs the source. This is the go/no-go.
3. When GPUs free: shard the 1.5k generation, then `verify_trajectories.py --check_step 16`
   as a hard gate before training. Local runs must be detached:
   `setsid bash -c 'CUDA_VISIBLE_DEVICES=N ... > run.log 2>&1' </dev/null &`
4. Train, run the go/no-go, record the numbers **here** before touching editing metrics.
