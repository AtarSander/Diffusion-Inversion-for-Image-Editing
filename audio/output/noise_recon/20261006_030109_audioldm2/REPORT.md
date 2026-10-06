# Noise normality vs reconstruction — audioldm2

- run: `/lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/code/lorainv/audio/outputs/noise_recon/audioldm2/main` (generated at git `08b32f305e`, scored at `08b32f305e`)
- N = 1024 clips (256 MedleyMD captions x 4 seeds), 100 inversion and denoising steps, CFG 1.0, 10.24 s, model `cvssp/audioldm2-large`
- LoRA (inversion pass only): `/lustre/pd03/hpc-tomtrz0116-1775130553/lstanisz/data/lorainv/checkpoints/attn_r8_a4_lr5e-5/checkpoint_step_4000.pt`
- Normality in latent space against the generation noise: KL = per-dimension Gaussian KL averaged over dimensions (x100); Corr = mean top-20 |Pearson| within 8x8 latent patches (Stable Audio: 8 frames x all channels). Gaussian Noise is a fresh draw, i.e. the null.
- Reconstruction on decoded audio against the generated source clip: mel MAE/PSNR/SSIM (audioldm_eval mel at 32 kHz, as in the editing benchmark) and LPAPS (CLAP, 10 s windows). Gaussian Noise reconstructs nothing: it is the unrelated-sample ceiling.

| Method | Corr ↓ | KL ×10² ↓ | MAE ↓ | LPAPS ↓ | PSNR ↑ | SSIM ↑ | noise std |
|---|---|---|---|---|---|---|---|
| Gaussian Noise | 0.1254 | 0.1963 | 0.1109 ± 0.0012 | 4.943 ± 0.012 | 17.21 ± 0.08 | 0.2164 ± 0.0020 | 1.0000 |
| DDIM Inv. | 0.1263 | 0.0205 | 0.0070 ± 0.0000 | 0.265 ± 0.003 | 39.72 ± 0.05 | 0.9849 ± 0.0002 | 0.9933 |
| DDIM Inv. + LoRA | 0.1257 | 0.0077 | 0.0043 ± 0.0000 | 0.124 ± 0.003 | 43.52 ± 0.06 | 0.9932 ± 0.0001 | 0.9960 |
| DDPM Inv. | 0.1251 | 0.1967 | 0.0087 ± 0.0000 | 0.386 ± 0.003 | 37.46 ± 0.01 | 0.9702 ± 0.0003 | 1.0000 |

Reference (generation noise) Corr = 0.1252: the finite-N floor every Corr above should be read against.

± is the standard error over clips. Per-clip values: `per_sample.csv`.
