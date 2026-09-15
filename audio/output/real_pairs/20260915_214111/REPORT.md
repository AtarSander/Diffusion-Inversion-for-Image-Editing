# H2: coarse- vs dense-trained adapter, paired vs the shared no-LoRA twin

Figure: `plots/dense_vs_coarse.png`. Bars are per-cell mean paired deltas, 95% CI.

| arm                      | cell   | metric      |   delta |    ci95 |
|:-------------------------|:-------|:------------|--------:|--------:|
| trajectory LoRA @4000    | w3.5   | lpaps       | -0.0385 | +0.0065 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w3.5   | clap        | -0.0011 | +0.0014 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w3.5   | muqt_sim_p0 | -0.0004 | +0.0013 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w3.5   | lpaps       | -0.0132 | +0.0080 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w3.5   | clap        | -0.0009 | +0.0019 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w3.5   | muqt_sim_p0 | -0.0002 | +0.0018 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w3.5   | lpaps       | -0.0624 | +0.0236 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w3.5   | clap        | -0.0000 | +0.0034 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w3.5   | muqt_sim_p0 | -0.0003 | +0.0051 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w3.5   | lpaps       | -0.1517 | +0.0770 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w3.5   | clap        | -0.0143 | +0.0081 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w3.5   | muqt_sim_p0 | -0.0210 | +0.0116 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w3.5   | lpaps       | -0.0803 | +0.0108 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w3.5   | clap        | +0.0050 | +0.0025 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w3.5   | muqt_sim_p0 | -0.0049 | +0.0038 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w3.5   | lpaps       | -0.2214 | +0.0633 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w3.5   | clap        | -0.0070 | +0.0140 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w3.5   | muqt_sim_p0 | +0.0004 | +0.0153 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w3.5   | lpaps       | -0.0769 | +0.0104 |
|                          | t99    |             |         |         |
| trajectory LoRA @4000    | w3.5   | clap        | +0.0053 | +0.0024 |
|                          | t99    |             |         |         |
| trajectory LoRA @4000    | w3.5   | muqt_sim_p0 | -0.0019 | +0.0042 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w3.5   | lpaps       | -0.2188 | +0.0673 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w3.5   | clap        | -0.0169 | +0.0138 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w3.5   | muqt_sim_p0 | -0.0125 | +0.0160 |
|                          | t99    |             |         |         |
| trajectory LoRA @4000    | w7.0   | lpaps       | -0.0303 | +0.0063 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w7.0   | clap        | -0.0019 | +0.0014 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w7.0   | muqt_sim_p0 | -0.0006 | +0.0012 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w7.0   | lpaps       | -0.0114 | +0.0089 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w7.0   | clap        | -0.0016 | +0.0018 |
|                          | t25    |             |         |         |
| real-audio forward-noise | w7.0   | muqt_sim_p0 | +0.0000 | +0.0018 |
|                          | t25    |             |         |         |
| trajectory LoRA @4000    | w7.0   | lpaps       | -0.0481 | +0.0165 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w7.0   | clap        | +0.0039 | +0.0028 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w7.0   | muqt_sim_p0 | +0.0025 | +0.0055 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w7.0   | lpaps       | -0.1682 | +0.0590 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w7.0   | clap        | -0.0008 | +0.0098 |
|                          | t50    |             |         |         |
| real-audio forward-noise | w7.0   | muqt_sim_p0 | -0.0017 | +0.0120 |
|                          | t50    |             |         |         |
| trajectory LoRA @4000    | w7.0   | lpaps       | -0.0660 | +0.0082 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w7.0   | clap        | +0.0017 | +0.0035 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w7.0   | muqt_sim_p0 | +0.0002 | +0.0049 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w7.0   | lpaps       | -0.1520 | +0.0484 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w7.0   | clap        | -0.0057 | +0.0085 |
|                          | t75    |             |         |         |
| real-audio forward-noise | w7.0   | muqt_sim_p0 | -0.0014 | +0.0123 |
|                          | t75    |             |         |         |
| trajectory LoRA @4000    | w7.0   | lpaps       | -0.0555 | +0.0098 |
|                          | t99    |             |         |         |
| trajectory LoRA @4000    | w7.0   | clap        | +0.0039 | +0.0028 |
|                          | t99    |             |         |         |
| trajectory LoRA @4000    | w7.0   | muqt_sim_p0 | +0.0004 | +0.0043 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w7.0   | lpaps       | -0.1588 | +0.0435 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w7.0   | clap        | -0.0052 | +0.0101 |
|                          | t99    |             |         |         |
| real-audio forward-noise | w7.0   | muqt_sim_p0 | -0.0044 | +0.0138 |
|                          | t99    |             |         |         |
