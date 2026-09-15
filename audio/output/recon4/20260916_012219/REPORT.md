# Four-arm real-audio reconstruction (chain-consistency gate)

Figure: `plots/recon4.png`. 35 MedleyDB tracks, t99, cfg 1.0, source caption.

|                              |   mel_psnr |   mel_ssim |   lpaps |   clap_src |
|:-----------------------------|-----------:|-----------:|--------:|-----------:|
| DDPM-inv                     |    23.3188 |     0.7091 |  2.8634 |     0.2255 |
| ODEInv, no LoRA              |    22.2165 |     0.6487 |  3.5073 |     0.2101 |
| ODEInv + trajectory LoRA     |    22.2981 |     0.6628 |  3.4305 |     0.2033 |
| ODEInv + real-fwd-noise LoRA |    15.5773 |     0.4411 |  5.3365 |     0.0760 |
