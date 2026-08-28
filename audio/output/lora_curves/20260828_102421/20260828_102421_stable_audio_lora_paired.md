# Stable Audio Open: inversion LoRA vs no LoRA -- MedleyMD hparam split, DDIM only

Generated 20260828_102421 from `/nas/lstanisz/code/lorainv/pwr-mount/audio/outputs/edits/medleymd/medleymd` at commit `7fc71a03c09f09b8299154507036eba6e26f1a8a`.
24 runs, 115 edits each. Every LoRA run is paired with the no-LoRA run at identical tstart and cfg_tar over the same rows.

Figure: `output/lora_curves/20260828_102421/plots/20260828_102421_stable_audio_lora_tradeoff.png`

## Paired differences, summarised per checkpoint

| checkpoint | metric | mean delta | worst-case CI | settings significant | settings better |
|---|---|---|---|---|---|
| saocos_r8_a4_lr5e-5 @4000 | LPAPS | -0.0573 | ±0.0234 | 8/8 | 8/8 |
| saocos_r8_a4_lr5e-5 @4000 | mel PSNR | +0.0596 | ±0.0488 | 7/8 | 6/8 |
| saocos_r8_a4_lr5e-5 @4000 | CLAP | +0.0021 | ±0.0035 | 5/8 | 5/8 |
| saocos_r8_a4_lr5e-5 @4000 EMA | LPAPS | -0.0568 | ±0.0230 | 8/8 | 8/8 |
| saocos_r8_a4_lr5e-5 @4000 EMA | mel PSNR | +0.0557 | ±0.0475 | 7/8 | 6/8 |
| saocos_r8_a4_lr5e-5 @4000 EMA | CLAP | +0.0019 | ±0.0034 | 5/8 | 5/8 |

## Every setting

| checkpoint | tstart | cfg_tar | metric | delta | 95% CI | p | rows better |
|---|---|---|---|---|---|---|---|
| saocos_r8_a4_lr5e-5 @4000 | 25 | 3.5 | lpaps | -0.0385 | ±0.0065 | 3.31e-21 | 105/115 |
| saocos_r8_a4_lr5e-5 @4000 | 25 | 7 | lpaps | -0.0303 | ±0.0062 | 2.33e-16 | 105/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 7 | lpaps | -0.0481 | ±0.0163 | 6.48e-08 | 102/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 3.5 | lpaps | -0.0624 | ±0.0234 | 7.50e-07 | 99/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 7 | lpaps | -0.0660 | ±0.0081 | 1.12e-30 | 109/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 3.5 | lpaps | -0.0803 | ±0.0107 | 3.74e-28 | 111/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 3.5 | lpaps | -0.0769 | ±0.0103 | 6.63e-28 | 111/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 7 | lpaps | -0.0555 | ±0.0097 | 5.40e-20 | 106/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 3.5 | lpaps | -0.0382 | ±0.0063 | 1.05e-21 | 105/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 7 | lpaps | -0.0302 | ±0.0061 | 1.01e-16 | 104/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 7 | lpaps | -0.0531 | ±0.0123 | 9.16e-14 | 103/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 3.5 | lpaps | -0.0615 | ±0.0230 | 7.33e-07 | 98/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 7 | lpaps | -0.0650 | ±0.0080 | 9.76e-31 | 110/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 3.5 | lpaps | -0.0783 | ±0.0103 | 1.91e-28 | 111/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 3.5 | lpaps | -0.0752 | ±0.0101 | 7.99e-28 | 111/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 7 | lpaps | -0.0526 | ±0.0099 | 2.98e-18 | 106/115 |
| saocos_r8_a4_lr5e-5 @4000 | 25 | 3.5 | psnr | -0.0892 | ±0.0100 | 5.20e-34 | 8/115 |
| saocos_r8_a4_lr5e-5 @4000 | 25 | 7 | psnr | -0.0783 | ±0.0091 | 1.31e-32 | 11/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 7 | psnr | +0.0835 | ±0.0412 | 1.24e-04 | 85/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 3.5 | psnr | +0.0180 | ±0.0488 | 4.72e-01 | 61/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 7 | psnr | +0.1340 | ±0.0336 | 2.81e-12 | 97/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 3.5 | psnr | +0.1395 | ±0.0358 | 7.10e-12 | 100/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 3.5 | psnr | +0.1287 | ±0.0361 | 1.96e-10 | 98/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 7 | psnr | +0.1410 | ±0.0303 | 2.94e-15 | 103/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 3.5 | psnr | -0.0905 | ±0.0102 | 5.78e-34 | 8/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 7 | psnr | -0.0797 | ±0.0093 | 9.77e-33 | 10/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 7 | psnr | +0.0784 | ±0.0405 | 2.35e-04 | 85/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 3.5 | psnr | +0.0087 | ±0.0475 | 7.21e-01 | 63/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 7 | psnr | +0.1277 | ±0.0368 | 5.02e-10 | 97/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 3.5 | psnr | +0.1351 | ±0.0348 | 8.32e-12 | 101/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 3.5 | psnr | +0.1290 | ±0.0320 | 2.03e-12 | 99/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 7 | psnr | +0.1367 | ±0.0300 | 8.95e-15 | 102/115 |
| saocos_r8_a4_lr5e-5 @4000 | 25 | 3.5 | clap | -0.0011 | ±0.0014 | 1.19e-01 | 54/115 |
| saocos_r8_a4_lr5e-5 @4000 | 25 | 7 | clap | -0.0019 | ±0.0014 | 1.02e-02 | 56/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 7 | clap | +0.0039 | ±0.0028 | 6.84e-03 | 70/115 |
| saocos_r8_a4_lr5e-5 @4000 | 50 | 3.5 | clap | -0.0000 | ±0.0033 | 9.84e-01 | 55/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 7 | clap | +0.0017 | ±0.0035 | 3.39e-01 | 63/115 |
| saocos_r8_a4_lr5e-5 @4000 | 75 | 3.5 | clap | +0.0050 | ±0.0025 | 1.56e-04 | 78/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 3.5 | clap | +0.0053 | ±0.0023 | 2.47e-05 | 77/115 |
| saocos_r8_a4_lr5e-5 @4000 | 99 | 7 | clap | +0.0039 | ±0.0027 | 6.08e-03 | 67/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 3.5 | clap | -0.0010 | ±0.0013 | 1.49e-01 | 55/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 25 | 7 | clap | -0.0018 | ±0.0014 | 1.19e-02 | 56/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 7 | clap | +0.0033 | ±0.0027 | 1.75e-02 | 71/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 50 | 3.5 | clap | -0.0001 | ±0.0034 | 9.49e-01 | 55/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 7 | clap | +0.0017 | ±0.0034 | 3.18e-01 | 61/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 75 | 3.5 | clap | +0.0049 | ±0.0024 | 1.64e-04 | 78/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 3.5 | clap | +0.0047 | ±0.0022 | 7.47e-05 | 78/115 |
| saocos_r8_a4_lr5e-5 @4000 EMA | 99 | 7 | clap | +0.0035 | ±0.0028 | 1.58e-02 | 71/115 |
