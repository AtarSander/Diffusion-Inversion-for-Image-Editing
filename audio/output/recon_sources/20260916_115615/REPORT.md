# Real-audio reconstruction, two sources, four arms (mel PSNR dB)

|                 |   MusicCaps (realfn training audio) |   MedleyDB (benchmark) |
|:----------------|------------------------------------:|-----------------------:|
| DDPM-inv        |                              23.686 |                 23.319 |
| ODEInv          |                              23.001 |                 22.216 |
| no LoRA         |                                     |                        |
| ODEInv +        |                              23.194 |                 22.298 |
| trajectory LoRA |                                     |                        |
| ODEInv +        |                              23.043 |                 15.577 |
| realfn LoRA     |                                     |                        |
