# MedleyMD editing baselines

696 edits per method (full `captions_gpt5.csv` split, 35 MusicDelta tracks), run on WCSS lem
(H100). Metrics from `editing/eval_medley.py`; per-example CSVs and JSON sit next to each run's
`audios/` under `audio/outputs/edits/medleymd/`.

Preservation is measured against **the input audio** — the `lower_bound_full` reference is a
per-row copy of that row's source mix, renamed `a{idx}.wav` so the paired metrics align by
filename.

> **The two models are evaluated separately and their numbers must not be pooled or ranked
> against each other.** See "Why the models cannot share a table" below — the reason is
> measured, not stylistic.

## AudioLDM2-large

200 steps, cfg_src 3.0 / cfg_tar 12.0, tstart 100 (DDIM 200). Output 16 kHz mono, source
truncated to 60 s.

| Method | LPAPS ↓ | CLAP ↑ | MuLan ↑ | mel PSNR ↑ | mel SSIM ↑ | FAD ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| DDPM-inv | **4.841** ± 0.652 | 0.354 ± 0.095 | 0.296 ± 0.149 | **18.572** | **0.576** | n/a |
| DDIM-inv | 6.070 ± 0.567 | 0.352 ± 0.110 | **0.343** ± 0.142 | 14.703 | 0.341 | n/a |
| SDEdit | 5.597 ± 0.547 | **0.357** ± 0.094 | 0.294 ± 0.142 | 18.032 | 0.425 | n/a |

**DDIM-inv vs DDPM-inv: −3.87 dB PSNR, +25% LPAPS, SSIM 0.341 vs 0.576.** That is the headroom
the inversion LoRA has to recover.

## Stable Audio Open 1.0

100 steps, cfg_src 1.0 / cfg_tar 3.5, tstart 50 (DDIM 100). Output 44.1 kHz stereo, capped near
47.5 s by the model's `sample_size`.

| Method | LPAPS ↓ | CLAP ↑ | MuLan ↑ | mel PSNR ↑ | mel SSIM ↑ | FAD ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| DDPM-inv | **3.502** ± 0.661 | 0.281 ± 0.109 | 0.188 ± 0.194 | **21.482** | **0.644** | n/a |
| DDIM-inv | 4.326 ± 0.673 | 0.285 ± 0.104 | 0.199 ± 0.170 | 17.048 | 0.530 | n/a |
| SDEdit | 6.193 ± 0.410 | **0.329** ± 0.091 | **0.231** ± 0.141 | 16.070 | 0.228 | n/a |

**DDIM-inv vs DDPM-inv: −4.43 dB PSNR, +24% LPAPS, SSIM 0.530 vs 0.644.**

## Why the models cannot share a table

Two systematic, measured asymmetries that have nothing to do with edit quality:

1. **Bandwidth.** AudioLDM2 synthesises at 16 kHz, so its Nyquist is 8 kHz and upsampling to the
   eval's 32 kHz cannot invent the band above it. Measured share of spectral energy above 8 kHz
   on the exact files the metrics read: AudioLDM2 edits **0.03%**, Stable Audio edits **0.14%**,
   source reference **0.16%**. Stable Audio keeps ~88% of the reference's high band, AudioLDM2
   ~19%, so any full-band spectral metric charges AudioLDM2 for missing bandwidth.
2. **Channels.** `MelPairedDataset` takes `audio[0:1]`. Stable Audio (stereo) is therefore
   compared left-against-left, while AudioLDM2 (mono) is compared as a downmix against the
   reference's left channel.

Within a model every method carries the identical handicap, so those comparisons — including
LoRA-DDIM vs DDIM — remain valid. Across models they are not, which is why the two tables above
are kept apart and never merged into one ranking.

## Sanity check that the numbers are trustworthy

**DDPM-inversion is best on every preservation metric in both tables.** That is required, not
discovered: `inversion_forward_process` does not invert — it draws `xts` directly from `x0` and
solves for the noise making the reverse step exact, so reconstruction is exact by construction.
LPAPS (CLAP-based, perceptual) and mel PSNR/SSIM (spectrogram-domain) are independent
measurements and agree on the ordering, which cross-validates both.

## The trade-off the LoRA must respect

The DDIM preservation gap is not free headroom. In AudioLDM2, DDIM-inv has the **best** MuLan
adherence of the three methods (0.343): it drifts further from the source, and some of that
drift lands on the target prompt. Improving preservation while losing adherence would be no
result at all, so both axes get reported together.

## FAD is unavailable, and it is not a configuration problem

`sqrtm(sigma1 @ sigma2)` returns complex with an imaginary component of **0.157**, far above the
1e-3 tolerance, so `FrechetAudioDistance` returns `-1`. The `eps` regularisation in
`calculate_frechet_distance` only triggers when `covmean` is non-finite, never when it is merely
complex, so it never applies.

Cause is the benchmark: MedleyMD is **35 unique excerpts**, so VGGish frame embeddings lie near
a low-dimensional manifold and the 128×128 covariance is close to singular. Deduplicating the
reference would not help — replicating samples leaves the covariance unchanged, so the 696-file
and 35-file references have identical covariance.

Options, undecided: report FAD as unavailable (current); force the eps offset whenever `covmean`
is complex, which yields a finite but poorly conditioned number; or drop FAD for this benchmark
and rely on the paired metrics, which are well-posed at n=696.

## Caveats when reading either table

- **Class balance is extreme**: GENRE 397, INSTR 195, MOOD 34, VOICE 31, OTHER 27, TEMPO 12.
  GENRE+INSTR is 85%, so each aggregate is effectively a GENRE score. Per-class numbers live in
  each run's `per_task_results.json`; treat the four small classes as directional.
- `alignment.py:344` still swallows per-file exceptions when building FAD/KL features without
  reporting a count. It does not affect LPAPS/CLAP/MuLan/PSNR/SSIM.
