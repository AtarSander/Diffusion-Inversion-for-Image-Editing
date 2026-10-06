# Noise normality vs reconstruction — stable_audio

- run: `/lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/code/lorainv/audio/outputs/noise_recon/stable_audio/main` (generated at git `08b32f305e`, scored at `08b32f305e`)
- N = 1024 clips (256 MedleyMD captions x 4 seeds), 100 inversion and denoising steps, CFG 1.0, 10.0 s, model `stabilityai/stable-audio-open-1.0`
- LoRA (inversion pass only): `/lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/data/lorainv/checkpoints/saocos_r8_a4_lr5e-5/checkpoint_step_4000.pt`
- Normality in latent space against the generation noise: KL = per-dimension Gaussian KL averaged over dimensions (x100); Corr = mean top-20 |Pearson| within 8x8 latent patches (Stable Audio: 8 frames x all channels). Gaussian Noise is a fresh draw, i.e. the null.
- Reconstruction on decoded audio against the generated source clip: mel MAE/PSNR/SSIM (audioldm_eval mel at 32 kHz, as in the editing benchmark) and LPAPS (CLAP, 10 s windows). Gaussian Noise reconstructs nothing: it is the unrelated-sample ceiling.

| Method | Corr ↓ | KL ×10² ↓ | MAE ↓ | LPAPS ↓ | PSNR ↑ | SSIM ↑ | noise std |
|---|---|---|---|---|---|---|---|
| Gaussian Noise | 0.1254 | 0.1949 | 0.1668 ± 0.0017 | 5.365 ± 0.024 | 13.76 ± 0.07 | 0.1948 ± 0.0022 | 0.9999 |
| DDIM Inv. | 0.1385 | 3.8711 | 0.0253 ± 0.0005 | 1.141 ± 0.023 | 29.82 ± 0.12 | 0.8588 ± 0.0036 | 0.8611 |
| DDIM Inv. + LoRA | 0.1383 | 3.2384 | 0.0187 ± 0.0004 | 0.773 ± 0.018 | 32.06 ± 0.11 | 0.9084 ± 0.0027 | 0.8761 |
| DDPM Inv. | 0.1253 | 0.1942 | 0.0174 ± 0.0001 | 0.677 ± 0.007 | 31.90 ± 0.03 | 0.9106 ± 0.0008 | 1.0000 |

Reference (generation noise) Corr = 0.1253: the finite-N floor every Corr above should be read against.

± is the standard error over clips. Per-clip values: `per_sample.csv`.
