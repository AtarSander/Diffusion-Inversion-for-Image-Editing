# MedleyMD editing baselines — AudioLDM2 & Stable Audio Open

696 edits per method (full `captions_gpt5.csv` split, 35 MusicDelta tracks), run on WCSS lem
(H100). Metrics from `editing/eval_medley.py`; per-example CSVs and JSON sit next to each run's
`audios/` under `audio/outputs/edits/medleymd/`.

Preservation is measured against **the input audio** — the `lower_bound_full` reference is a
per-row copy of that row's source mix, renamed `a{idx}.wav` so the paired metrics align by
filename.

## Results

| Method | LPAPS ↓ | CLAP ↑ | MuLan ↑ | mel PSNR ↑ | mel SSIM ↑ | FAD ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| AudioLDM2 DDPM-inv | 4.841 ± 0.652 | 0.354 ± 0.095 | 0.296 ± 0.149 | 18.572 | 0.576 | n/a |
| AudioLDM2 DDIM-inv | 6.070 ± 0.567 | 0.352 ± 0.110 | **0.343** ± 0.142 | 14.703 | 0.341 | n/a |
| AudioLDM2 SDEdit | 5.597 ± 0.547 | **0.357** ± 0.094 | 0.294 ± 0.142 | 18.032 | 0.425 | n/a |
| StableAudio DDPM-inv | **3.502** ± 0.661 | 0.281 ± 0.109 | 0.188 ± 0.194 | **21.482** | **0.644** | n/a |
| StableAudio DDIM-inv | 4.326 ± 0.673 | 0.285 ± 0.104 | 0.199 ± 0.170 | 17.048 | 0.530 | n/a |
| StableAudio SDEdit | 6.193 ± 0.410 | 0.329 ± 0.091 | 0.231 ± 0.141 | 16.070 | 0.228 | n/a |

Config: AudioLDM2-large 200 steps, cfg_src 3.0 / cfg_tar 12.0, tstart 100 (DDIM 200);
Stable Audio Open 100 steps, cfg_src 1.0 / cfg_tar 3.5, tstart 50 (DDIM 100). DDIM uses
`tstart == steps`, since partial inversion is not DDIM inversion.

## Sanity check that the numbers are trustworthy

**DDPM-inversion is best on every preservation metric, on both models.** That is required, not
discovered: `inversion_forward_process` does not invert — it draws `xts` directly from `x0` and
solves for the noise that makes the reverse step exact, so reconstruction is exact by
construction. LPAPS (CLAP-based, perceptual) and mel PSNR/SSIM (spectrogram-domain) are
independent measurements and agree on the ordering, which cross-validates both.

## The headline for this project

DDIM-inversion is much worse at preserving the input than DDPM-inversion, and that gap is
exactly what the inversion LoRA is meant to close:

| | LPAPS ↓ | mel PSNR ↑ | mel SSIM ↑ |
| --- | --- | --- | --- |
| AudioLDM2: DDIM vs DDPM | 6.070 vs 4.841 (+25%) | 14.70 vs 18.57 (**−3.87 dB**) | 0.341 vs 0.576 |
| StableAudio: DDIM vs DDPM | 4.326 vs 3.502 (+24%) | 17.05 vs 21.48 (**−4.43 dB**) | 0.530 vs 0.644 |

The honest framing stays **LoRA-DDIM vs plain DDIM inversion**, with DDPM-inversion as the
reference ceiling on fidelity — not "we beat everything".

The trade-off is real and must not be ignored: AudioLDM2 DDIM has the **best** MuLan adherence
(0.343) of any method here. It drifts further from the source, and some of that drift moves it
toward the target prompt. So the LoRA has to improve preservation *without* surrendering
adherence; reporting LPAPS/PSNR alone would be misleading.

## FAD is unavailable, and it is not a configuration problem

`sqrtm(sigma1 @ sigma2)` returns complex with an imaginary component of **0.157**, far above
the 1e-3 tolerance, so `FrechetAudioDistance` returns `-1`. The `eps` regularisation in
`calculate_frechet_distance` only triggers when `covmean` is non-finite, never when it is merely
complex, so it never applies.

Cause is the benchmark: MedleyMD is **35 unique excerpts**, so VGGish frame embeddings lie near
a low-dimensional manifold and the 128×128 covariance is close to singular. Deduplicating the
reference would not help — replicating samples leaves the covariance unchanged, so the 696-file
and 35-file references have identical covariance.

Options, undecided: report FAD as unavailable (current); force the eps offset whenever `covmean`
is complex, which yields a finite but poorly conditioned number; or drop FAD for this benchmark
and rely on the paired metrics, which are well-posed at n=696.

## Caveats when reading the table

- **Class balance is extreme**: GENRE 397, INSTR 195, MOOD 34, VOICE 31, OTHER 27, TEMPO 12.
  GENRE+INSTR is 85%, so the aggregate is effectively a GENRE score. Per-class numbers live in
  each run's `per_task_results.json`; treat the four small classes as directional.
- Cross-model comparison is loose. Stable Audio outputs 44.1 kHz stereo capped near 47.5 s,
  AudioLDM2 16 kHz mono truncated to 60 s. Compare methods within a model.
- `alignment.py:344` still swallows per-file exceptions when building FAD/KL features without
  reporting a count. It does not affect LPAPS/CLAP/MuLan/PSNR/SSIM.
