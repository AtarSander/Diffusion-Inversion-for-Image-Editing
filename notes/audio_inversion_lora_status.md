# Audio Inversion LoRA — Status & Open Items

Porting the shifted-denoiser inversion LoRA (see [inversion_methods.md](inversion_methods.md))
from SDXL/SD1.5 to **AudioLDM2**, to be compared against the DDPM-inversion / DDIM / SDEdit
baselines in `audio/editing/` on the MedleyMD editing benchmark.

Most recent first. Keep this file current — it is the handover doc between sessions.

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
