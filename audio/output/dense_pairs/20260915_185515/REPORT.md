# H2: coarse- vs dense-trained adapter, paired vs the shared no-LoRA twin

Figure: `plots/dense_vs_coarse.png`. Bars are per-cell mean paired deltas, 95% CI.

| arm                            | cell   | metric      |   delta |    ci95 |
|:-------------------------------|:-------|:------------|--------:|--------:|
| coarse (100-step trajectories) | w3.5   | lpaps       | -0.0385 | +0.0065 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w3.5   | clap        | -0.0011 | +0.0014 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w3.5   | muqt_sim_p0 | -0.0004 | +0.0013 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w3.5   | lpaps       | -0.0390 | +0.0065 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w3.5   | clap        | -0.0011 | +0.0014 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w3.5   | muqt_sim_p0 | -0.0004 | +0.0013 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w3.5   | lpaps       | -0.0624 | +0.0236 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w3.5   | clap        | -0.0000 | +0.0034 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w3.5   | muqt_sim_p0 | -0.0003 | +0.0051 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w3.5   | lpaps       | -0.0669 | +0.0228 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w3.5   | clap        | +0.0001 | +0.0033 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w3.5   | muqt_sim_p0 | -0.0008 | +0.0046 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w3.5   | lpaps       | -0.0803 | +0.0108 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w3.5   | clap        | +0.0050 | +0.0025 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w3.5   | muqt_sim_p0 | -0.0049 | +0.0038 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w3.5   | lpaps       | -0.0805 | +0.0109 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w3.5   | clap        | +0.0047 | +0.0025 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w3.5   | muqt_sim_p0 | -0.0048 | +0.0039 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w3.5   | lpaps       | -0.0769 | +0.0104 |
|                                | t99    |             |         |         |
| coarse (100-step trajectories) | w3.5   | clap        | +0.0053 | +0.0024 |
|                                | t99    |             |         |         |
| coarse (100-step trajectories) | w3.5   | muqt_sim_p0 | -0.0019 | +0.0042 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w3.5   | lpaps       | -0.0787 | +0.0103 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w3.5   | clap        | +0.0051 | +0.0024 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w3.5   | muqt_sim_p0 | -0.0028 | +0.0042 |
|                                | t99    |             |         |         |
| coarse (100-step trajectories) | w7.0   | lpaps       | -0.0303 | +0.0063 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w7.0   | clap        | -0.0019 | +0.0014 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w7.0   | muqt_sim_p0 | -0.0006 | +0.0012 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w7.0   | lpaps       | -0.0309 | +0.0062 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w7.0   | clap        | -0.0019 | +0.0014 |
|                                | t25    |             |         |         |
| dense (991-step trajectories)  | w7.0   | muqt_sim_p0 | -0.0007 | +0.0012 |
|                                | t25    |             |         |         |
| coarse (100-step trajectories) | w7.0   | lpaps       | -0.0481 | +0.0165 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w7.0   | clap        | +0.0039 | +0.0028 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w7.0   | muqt_sim_p0 | +0.0025 | +0.0055 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w7.0   | lpaps       | -0.0543 | +0.0135 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w7.0   | clap        | +0.0032 | +0.0028 |
|                                | t50    |             |         |         |
| dense (991-step trajectories)  | w7.0   | muqt_sim_p0 | +0.0011 | +0.0055 |
|                                | t50    |             |         |         |
| coarse (100-step trajectories) | w7.0   | lpaps       | -0.0660 | +0.0082 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w7.0   | clap        | +0.0017 | +0.0035 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w7.0   | muqt_sim_p0 | +0.0002 | +0.0049 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w7.0   | lpaps       | -0.0667 | +0.0085 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w7.0   | clap        | +0.0018 | +0.0035 |
|                                | t75    |             |         |         |
| dense (991-step trajectories)  | w7.0   | muqt_sim_p0 | +0.0005 | +0.0049 |
|                                | t75    |             |         |         |
| coarse (100-step trajectories) | w7.0   | lpaps       | -0.0555 | +0.0098 |
|                                | t99    |             |         |         |
| coarse (100-step trajectories) | w7.0   | clap        | +0.0039 | +0.0028 |
|                                | t99    |             |         |         |
| coarse (100-step trajectories) | w7.0   | muqt_sim_p0 | +0.0004 | +0.0043 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w7.0   | lpaps       | -0.0552 | +0.0100 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w7.0   | clap        | +0.0040 | +0.0028 |
|                                | t99    |             |         |         |
| dense (991-step trajectories)  | w7.0   | muqt_sim_p0 | +0.0010 | +0.0043 |
|                                | t99    |             |         |         |
