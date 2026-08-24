# Session notes — timestep-embedding LoRA, and the edit-geometry reconstruction eval

**Date:** 2026-08-20 → 2026-08-21
**Conversation ID:** `cb1a256d-aed0-4ac2-b386-49c65bbbb606`
**Session:** https://claude.ai/code/session_01VtLGeWNuakTRHKN5PtpRE9
**Commits:** `990062b`, `0008bc4`, `d276a74`, `8e0836d`, `781e6b6`, `c431e9e` on `audio_edit`
(interleaved with the Stable Audio port, `68428ee` / `f180434` / `2f1f4f2`, which is separate work)

Continues [session_2026-08-20_editing_sweep.md](session_2026-08-20_editing_sweep.md). One
experiment (`q4_fullte_r32_a16_lr5e-4`, array index 27, slurm 5739051), one eval change that made
it measurable at the geometry the edits use, and three corrections to claims made along the way.

---

## 1. Hypothesis

Every adapter so far left the timestep-embedding path frozen, so nothing the LoRA touched saw `t`
directly. But the shift gap it has to close is ~300x larger at `t <= 250` than at the noisy end,
i.e. the residual's dominant structure is a function of `t`. The timestep embedding is where the
schedule enters the network, so it should be the highest-leverage place to spend parameters —
24 modules against the 1467 the `full` preset already covered.

Argued against at the time, on the grounds that the exact correction is a Jacobian-vector product
(spatially varying) while a time-embedding LoRA only produces a spatially uniform FiLM modulation.
That objection turned out to be the wrong worry: the adapter fit the objective *better* than
anything before. The right worry, unstated, was stability.

## 2. Method

`full` preset extended with `time_embedding.linear_1`, `time_embedding.linear_2` and the 22
per-ResNet `time_emb_proj`: **1467 -> 1491 modules**. Verified on a meta-device UNet that the three
presets resolve to 1024 / 1280 / 1491 and that the new dotted names match exactly one module each
with no collision against `class_embedding` (`class_embed_type` is `None`).

Index 27 re-runs index 26 (q4, r32/a16, lr 5e-4) under the new preset, capped at 6000 steps to
line up with the step-6000 checkpoint the scored sweep used, checkpointing and evaluating
reconstruction every 1000. Run name `q4_fullte_*`, deliberately not `q4_full_*`: reusing the name
would have written over `checkpoint_step_6000.pt`, which `lora_sweep_configs.sh` still points at.

**Eval change — real audio at the edit geometry.** The adapter trains on 10.24 s trajectories and
is deployed on edits that `create_truncated_audio` feeds at up to 60 s, and nothing had measured
whether the correction survives that. The in-training reconstruction now takes each track at its
own length capped at 60 s, from the start, which is exactly what the edit pipeline sees.

## 3. Results

Figure and tables: `audio/output/time_emb_lora/20260821_115246/`. Regenerate with
`uv run editing/plot_training_curves.py --log_path <slurm .err>`.

**The objective: best fit yet.** Val loss 3.543e-6 at step 1000 = **92.7% of the shift gap
closed**, breaking the 85-88% plateau that had held across every rank, learning rate and module
preset — and reaching it in 1000 steps rather than 6000. Best EMA is step 3000 (3.408e-6, 92.9%).
The hypothesis was right about leverage.

**Reconstruction: negative.** Paired against plain DDIM on identical fixtures, real-audio mel PSNR
peaks at step 1000 (+0.62 dB) and is *below* baseline everywhere after, bottoming at -2.32 dB at
step 4000. Val loss rises monotonically after step 1000, so the true optimum is at or before the
first eval and was never measured.

The generated set is the clean control: its geometry did not change, and it shows the same shape
(+1.27 dB at step 1000, -2.57 dB at 4000). So this is not a long-window artifact — the adapter
genuinely damages the round trip at lr 5e-4. Perturbing the global noise-level conditioning is
high-leverage in both directions.

**Dose-response fails again.** 87.6% -> 92.7% on the objective bought +0.62 dB of reconstruction.
That is the fifth independent measurement pointing the same way; see the null-result summary in
[audio_inversion_lora_status.md](audio_inversion_lora_status.md).

**The train/deploy geometry hypothesis is dead.** Plain DDIM reconstructs real audio at the edit
geometry at **37.31 dB** mel PSNR, *better* than the 34.42 dB it gets on fixed 10.24 s crops. There
is no geometry cliff, so a mismatch between the 10.24 s training window and the up-to-60 s edit
window cannot explain why the adapter does not transfer.

## 4. Things that are easy to get wrong

**A fixed 60 s crop does not exist on this pool.** Only 14 of the 35 MedleyDB mixes reach 60 s
(13.1-302.8 s, median 49.4), so `recon_duration_s=60` would have tripped the
`mel.shape[2] > frames` assert during fixture prep. Natural lengths give 13.1-60 s windows, mean
43.6 s. `create_truncated_audio` caps and never pads, so the benchmark was never uniformly 60 s
either — `[8, 1500, 16]` is its worst case, not its typical one.

**Distinct real crops cap at one per track.** A natural-length window leaves no offset to vary, so
`recon_num_real` above the pool size yields byte-identical duplicates. That is worse than wasteful:
identical rows correlate at exactly 1.0, so `corr_topk` and `kl_per_dim` go degenerate. Index 27
asks for 35. At n=35 those metrics sit at their floor anyway (0.627 measured against a 0.624
reference), so reconstruction is the only signal on the real set.

**Batching buys nothing at these lengths.** Measured on an A5000, per-sample forward time at 60 s
is flat in batch size (0.649 / 0.646 / 0.639 / 0.644 s at batch 1/2/4/8) because one 60 s latent
already saturates the GPU; at 10.24 s batch 8 is 2.9x more efficient per sample. So letting
variable-shaped crops fall back to batch 1 costs nothing, and `batch_latents` only groups
same-shaped neighbours. Memory is never the constraint either: 0.19 GiB of activations per sample
at 60 s, 7.2 GiB total at batch 8.

**The UNet does not need a power-of-two time extent.** All 19 distinct latent heights the pool
produces forward cleanly, including odd ones like 327 and 1475.

**The distributional checks need one shape.** `corr_topk` and `kl_per_dim` estimate per-dimension
statistics across the batch, which variable-length crops do not admit. `crop_to_window` cuts every
latent to the generated set's window — the shortest geometry in play — which also keeps those
numbers comparable with earlier runs.

**Pick checkpoints on reconstruction, and pick each arm at its own step.** Raw weights are best at
1000; the EMA at 1000 is its *worst* point (79.9% closed — the average has not caught up) and
bottoms at 3000. Pairing both arms at one step would spend 8 sweep runs on a checkpoint already
known to be bad, so the sweep scores `checkpoint_step_1000.pt` and `checkpoint_step_3000_ema.pt`.

## 5. Corrections made this session

**Mel PSNR is not comparable between the two eval paths**, and the reason took three tries to get
right. Final, verified account, now in `reconstruct.py`'s `mel_metrics` docstring:

- *What is compared.* The in-training eval compares two VAE decodes of a latent, so decoder error
  cancels and no vocoder runs. `audioldm_eval`'s `MelPairedDataset` pairs two directories of **wav
  files**, so the benchmark scores the original audio against the written output and carries VAE
  encode+decode, the 16 kHz HiFi-GAN vocoder and an STFT re-analysis. This is the larger of the two
  differences.
- *How the mel is scaled.* The benchmark re-analyses each wav (first channel only, 16 kHz, DC
  removed) into `clip((20 * log10(mel) + 80) / 100, 0, 1)`, mapping -80..+20 dB onto [0, 1].
- *`data_range` is not a bug.* Those mels live in [0, 1] by construction, so scikit-image's float
  fallback of 1.0 is the representation's true full scale — the conventional choice, as 255 is for
  uint8. `mel_metrics` here uses a per-file empirical range (~12.6 on VAE-decoded log-mels), which
  is the less standard of the two. `alignment.py` needs no fix.

Two intermediate claims were wrong and are retracted: that `data_range` alone explained the
37.31 vs 19.50 dB gap (~16 dB was computed by changing the range while holding MSE fixed, but the
benchmark's /100 moves MSE too), and that `alignment.py`'s bare `psnr()` call was a bug.

**Also worth knowing:** the benchmark truncates both mels to the shorter clip from frame 0, so its
numbers assume the output is sample-aligned with the input.

## 6. Editing benchmark: the sixth negative (2026-08-24)

Sweep indices 48-63 ran as job `5746514` (16/16 edits, Aug 21) and scored as `5756175` (16/16,
Aug 24). Paired against no-LoRA twins at identical tstart and cfg_tar over the same 115 rows:
`audio/output/lora_curves/20260824_065600/`.

| checkpoint | LPAPS | worst CI | mel PSNR | CLAP |
|---|---|---|---|---|
| `q4_fullte @1000` | **+0.0035** | ±0.0115 | -0.0223 | -0.0005 |
| `q4_fullte @3000 EMA` | **+0.0059** | ±0.0160 | -0.0209 | -0.0014 |
| scale: DDPM-inv | -0.1383 | | -0.1765 | +0.0019 |
| scale: SDEdit | +0.7711 | | -0.9267 | +0.0106 |

Negative LPAPS is better preservation, so both new checkpoints are on the **wrong side**, by an
amount ~2.5% of DDPM-inv's shift and ~0.1% of the front's 2.9 LPAPS span. Every confidence
interval is ~3x the mean delta and "settings better" is 5/8, i.e. a coin flip: the honest reading
is indistinguishable from zero with a hint of slightly worse. On the figure all eight LoRA
checkpoints sit on top of the black no-LoRA front while DDPM-inv and SDEdit trace visibly
different curves.

**The dose-response is now inverted, not just absent.** These two are the largest-magnitude
deltas of all eight checkpoints on both LPAPS (+0.0035, +0.0059 against ±0.0025 for the rest) and
mel PSNR (-0.022, -0.021 against ±0.013) — and both in the wrong direction. The adapter that fits
the objective best is the one that hurts editing most.

That is six independent measurements. **Inversion fidelity is closed as a lever.**

## 7. Next steps

1. **Not recommended:** chasing the pre-step-1000 peak with `recon_every_steps=250`. It would
   sharpen the best-case number, but the ceiling is now bounded by six measurements, and the
   step-1000 point was only +0.62 dB on reconstruction and negative on the benchmark.
3. **The open direction remains target-side**, not inversion: guidance schedule and
   cross-attention control. Adapter capacity (rank), module coverage (attn / ff / conv), schedule
   restriction (q4) and now the timestep-embedding path have all been tried.
