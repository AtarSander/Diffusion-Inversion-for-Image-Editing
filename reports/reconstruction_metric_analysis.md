# Reconstruction/background-preservation analysis

## Scope

This analysis compares the completed SD1.5 `cycle_lambda=1`, rank-16
LoRA+DirectInv run with the reproduced SD1.5 DirectInv control, paired by all
700 PIE-Bench image IDs and separately for P2P, PnP, MasaCtrl, and
Pix2Pix-Zero. The standard uniform rank-16 LoRA+DirectInv results were also
checked as a robustness control. Their paired gaps are nearly the same, so the
findings below are not a special property of the cycle-loss checkpoint.

The reproduced SD1.5 control is the primary baseline because it shares the
model and evaluator with the LoRA runs. The paper means are included as a
secondary reference. In the paper, PnP uses SD1.5, while P2P, MasaCtrl, and
Pix2Pix-Zero use SD1.4.

For every paired gap below, a positive number means that LoRA+DirectInv is
worse. For PSNR and SSIM, the gap is `baseline - experiment`; for Structure,
LPIPS, and MSE, it is `experiment - baseline`.

## Main conclusion

The evidence does **not** support a single general “LoRA inversion reconstructs
poorly” failure.

1. P2P is not failing against the matched SD1.5 DirectInv baseline. It improves
   every reconstruction/background metric, usually on about 90% of images.
2. PnP has a small but systematic regression. MasaCtrl has the only large,
   broad regression. Pix2Pix-Zero has a modest mean regression dominated by
   catastrophic outliers.
3. For MasaCtrl and Pix2Pix-Zero, the saved source/reconstruction branch is
   essentially at the VAE reconstruction floor for both methods. The loss
   appears in the edited target branch. This is direct evidence of an
   editor--trajectory interaction, not a failure to recover the source image.
4. The lost pixels are not concentrated at the edit-mask boundary. About 55%
   of positive excess squared error is farther than 32 pixels from the mask.
   PnP and MasaCtrl lose disproportionately on contours and textured details.
5. Metric-mask artifacts materially change absolute values, especially MSE and
   SSIM, but they do not explain the sign of the paired PnP/MasaCtrl/Pix2Pix-Zero
   regressions.
6. MasaCtrl has an additional implementation confound: classic DirectInv
   inverts with an empty prompt, whereas the fused path inverts with the source
   prompt and then computes offsets using empty/target contexts. This is not a
   controlled replacement of only the inversion operator.

## Paper, reproduced control, and representative LoRA result

All Structure and LPIPS values are multiplied by 1,000; MSE by 10,000; SSIM by
100.

| Editor | Result | Structure down | PSNR up | LPIPS down | MSE down | SSIM up |
|---|---|---:|---:|---:|---:|---:|
| P2P | Paper DirectInv (SD1.4) | 11.65 | 27.22 | 54.55 | 32.86 | 84.76 |
| P2P | Reproduced DirectInv (SD1.5) | 31.35 | 22.57 | 125.51 | 91.27 | 78.54 |
| P2P | LoRA+DirectInv | 14.45 | 25.92 | 74.51 | 45.37 | 83.37 |
| P2P | Paired gap vs reproduced | -16.90 | -3.35 | -50.99 | -45.90 | -4.83 |
| PnP | Paper DirectInv (SD1.5) | 24.29 | 22.46 | 106.06 | 80.45 | 79.68 |
| PnP | Reproduced DirectInv (SD1.5) | 23.33 | 22.46 | 105.51 | 79.97 | 80.22 |
| PnP | LoRA+DirectInv | 24.38 | 22.36 | 110.28 | 81.87 | 79.88 |
| PnP | Paired gap vs reproduced | +1.05 | +0.10 | +4.76 | +1.90 | +0.34 |
| MasaCtrl | Paper DirectInv (SD1.4) | 24.70 | 22.64 | 87.94 | 81.09 | 81.33 |
| MasaCtrl | Reproduced DirectInv (SD1.5) | 23.57 | 22.82 | 86.83 | 79.19 | 82.02 |
| MasaCtrl | LoRA+DirectInv | 28.49 | 21.89 | 98.91 | 98.95 | 80.50 |
| MasaCtrl | Paired gap vs reproduced | +4.92 | +0.93 | +12.09 | +19.76 | +1.52 |
| Pix2Pix-Zero | Paper DirectInv (SD1.4) | 49.22 | 21.53 | 138.98 | 127.32 | 77.05 |
| Pix2Pix-Zero | Reproduced DirectInv (SD1.5) | 49.99 | 21.47 | 142.46 | 133.49 | 77.49 |
| Pix2Pix-Zero | LoRA+DirectInv | 52.16 | 21.22 | 147.00 | 140.33 | 76.95 |
| Pix2Pix-Zero | Paired gap vs reproduced | +2.17 | +0.25 | +4.55 | +6.84 | +0.54 |

The paper and reproduced values agree closely for PnP and MasaCtrl and
reasonably for Pix2Pix-Zero. P2P is the exception: the reproduced SD1.5
DirectInv control is much worse than the paper's SD1.4 result on all five
preservation metrics. Therefore the P2P paper gap must not be interpreted as a
LoRA-specific regression. Against the matched reproduction, LoRA recovers most
of that gap.

Paired bootstrap 95% confidence intervals for the mean worse-direction gap are:

| Editor | Structure | PSNR | LPIPS | MSE | SSIM |
|---|---:|---:|---:|---:|---:|
| P2P | [-18.60, -15.28] | [-3.62, -3.08] | [-55.74, -46.47] | [-51.02, -41.06] | [-5.28, -4.40] |
| PnP | [0.47, 1.62] | [0.03, 0.16] | [3.69, 5.85] | [0.99, 2.84] | [0.22, 0.46] |
| MasaCtrl | [3.91, 5.93] | [0.77, 1.10] | [9.70, 14.66] | [16.01, 23.78] | [1.27, 1.79] |
| Pix2Pix-Zero | [0.35, 3.97] | [0.11, 0.40] | [1.36, 7.78] | [0.92, 12.96] | [0.21, 0.86] |

## What the evaluator actually scores

The output files contain four horizontal panels: instruction, source,
source/reconstruction branch, and edited result. The official evaluator always
crops the **rightmost** panel. Consequently, these are not metrics on a pure
reconstruction image:

- Structure compares the full source image with the full edited result.
- PSNR, LPIPS, MSE, and SSIM compare the source with the edited result after
  masking out the annotated region that was expected to change.

The paper accurately describes them as structure retention and background
preservation. Calling all five “reconstruction metrics” obscures the most
important distinction in this analysis: an inversion may recover the source
branch very well and still interact badly with an editor's target branch.

PIE masks are hand annotations of the intended edit region. The evaluator also
forcibly marks the entire one-pixel image border as edited. Of 700 samples, 144
have no remaining unedited pixels and become NaN for PSNR, LPIPS, MSE, and
SSIM. This leaves 556 valid samples. The exclusions are highly non-uniform:

- style change: 78 of 80 excluded;
- object addition: 27 of 80 excluded;
- random: 19 of 140 excluded;
- background change: 10 of 80 excluded.

Thus Structure uses 700 images, while every masked metric uses a different
556-image task mixture that almost entirely omits global style changes.

## Metric-by-metric audit

### 1. Structure Distance

Implementation:

1. Resize each full image to 224 pixels.
2. Extract last-layer key tokens from DINO ViT-B/8.
3. Construct the pairwise cosine-similarity matrix between all tokens.
4. Return the MSE between the source and edited-image similarity matrices.

Lower is better. No edit mask is used. The metric is sensitive to object
layout, silhouette, pose, and large geometric changes, but it also penalizes a
successful intended change when that change alters global structure. It is
therefore the most independent of the four background metrics and the least
pure measure of reconstruction.

There is an inherited upstream implementation anomaly: input pixels remain in
the 0--255 range before the standard ImageNet normalization intended for 0--1
inputs. This may affect the semantic meaning of the absolute values. It does
not selectively favor the baseline because all paper-style evaluations use
the same path, but a corrected `/255` sensitivity run should accompany any
claim based specifically on DINO Structure.

Measured loss:

- P2P improves by 16.90 and wins on 88.6% of images.
- PnP loses 1.05 and wins on only 37.3%.
- MasaCtrl loses 4.92 and wins on 35.9%.
- Pix2Pix-Zero loses 2.17 and wins on 43.7%.

The top 10% of positive failures account for 59% of PnP's, 51% of MasaCtrl's,
and 63% of Pix2Pix-Zero's total positive Structure gap. PnP's largest
Structure failures are style edits such as stained glass and pen-and-ink,
where the intended global transformation itself changes the DINO relation
map. MasaCtrl's largest average category losses are color, material, and the
random set. Pix2Pix-Zero's are material and object addition.

### 2. MSE outside the edit mask

The official implementation sets the edited region to black in both images,
then takes the mean squared error over the **entire** 512 by 512 by 3 tensor.
If `q` is the unedited fraction, then:

`official_MSE = q * true_mean_MSE_over_unedited_pixels`

This is an area-dilution artifact. Images with a small unedited region receive
artificially small MSE. It also changes the weighting of images in the dataset
mean. Since each baseline/experiment pair uses the same mask, it does not
reverse the per-image direction, but it understates the magnitude of failures
on large-mask examples.

The independent pixel recomputation matches the official CSVs within 3.5e-8.
After correctly normalizing by the number of unedited pixels, the means are:

| Editor | DirectInv | LoRA+DirectInv | Proper MSE gap |
|---|---:|---:|---:|
| P2P | 146.35 | 73.04 | -73.31 |
| PnP | 129.54 | 131.74 | +2.20 |
| MasaCtrl | 125.29 | 158.39 | +33.10 |
| Pix2Pix-Zero | 200.36 | 210.67 | +10.31 |

The official gaps were -45.90, +1.90, +19.76, and +6.84 respectively. Correct
mask normalization therefore makes the MasaCtrl and Pix2Pix-Zero deficits
larger, not smaller.

Where the excess squared error occurs:

| Editor | 0--4 px from mask | 5--16 px | 17--32 px | farther than 32 px | edge-detail gap | smooth-region gap |
|---|---:|---:|---:|---:|---:|---:|
| PnP | +3.01 | +1.69 | +1.84 | +3.47 | +5.28 | +0.68 |
| MasaCtrl | +36.95 | +31.74 | +24.26 | +28.72 | +43.17 | +16.66 |
| Pix2Pix-Zero | +13.92 | +10.33 | +7.12 | +10.49 | +11.25 | +8.82 |

About 55% of positive excess error is farther than 32 pixels from the mask for
all three editors. This rules out a mask-boundary-only explanation. Edges and
fine details carry 59% of PnP's, 53% of MasaCtrl's, and 47% of Pix2Pix-Zero's
positive excess pixel error. The spatial maps place PnP loss mainly on
lower/central objects, MasaCtrl loss strongly on central subjects, and
Pix2Pix-Zero loss more irregularly across the frame.

Global RGB bias explains essentially none of PnP's gap, 1.4% of MasaCtrl's,
and 15.5% of Pix2Pix-Zero's. The main problem is shape, identity, contour, and
texture redraw rather than a global exposure or color shift.

The largest properly normalized MSE failures are concrete semantic spillovers:

- PnP: green-hat edit changes hair/clothing outside the hat (+157.2); a gold to
  stone Buddha edit alters unmasked tone/detail (+133.4); a blue-background cat
  edit redraws unmasked fur and facial detail (+95.3).
- MasaCtrl: road-to-snow redraws the couple's bodies/pose (+663.5); adding a
  field to a painting changes the people (+486.5); removing latte art changes
  the entire cup layout (+461.0); adding a house changes the girl's identity
  and rendering style (+447.3).
- Pix2Pix-Zero: adding a chain/star to a cat changes the whole composition
  (+983.2); deer-to-iron changes geometry/style (+493.9); napkin-to-pizza
  regenerates the table (+489.6); a pose edit replaces the figure (+372.7).

### 3. PSNR outside the edit mask

For every individual output, PSNR is exactly:

`PSNR = -10 * log10(official_MSE)`

The maximum numerical discrepancy from this identity in the saved CSVs is
negligible. PSNR is therefore not independent evidence from MSE. The mask
dilution adds `-10 * log10(q)` dB to the proper unedited-pixel PSNR. It inflates
the absolute values, especially for large masks.

Because `q` is identical within each paired comparison, the paired PSNR gap is
unchanged by correcting the area normalization. The corrected absolute means
are 20.40 versus 23.75 for P2P, 20.29 versus 20.19 for PnP, 20.66 versus 19.72
for MasaCtrl, and 19.30 versus 19.05 for Pix2Pix-Zero. The reported paired
losses remain 0.10 dB for PnP, 0.93 dB for MasaCtrl, and 0.25 dB for
Pix2Pix-Zero.

### 4. LPIPS outside the edit mask

The evaluator blackens the edited region in both images, maps the resulting
full image to [-1, 1], and runs TorchMetrics LPIPS with a SqueezeNet backbone.
Lower is better.

This is not a spatially normalized masked LPIPS. The large identical black
region contributes identical activations, while the sharp artificial boundary
affects convolutional receptive fields on both sides of the mask. There is no
simple division by `q` that corrects it. The paired use of the same mask makes
comparison safer than the absolute score, but the metric still changes its
effective weighting with mask shape and size.

The paired gaps are -50.99 for P2P, +4.76 for PnP, +12.09 for MasaCtrl, and
+4.55 for Pix2Pix-Zero. LPIPS shows the same broad MasaCtrl failure as MSE but
also detects perceptual changes that pixel MSE treats differently. Examples
include altered texture/semantics without a large global color shift.

PnP's largest average LPIPS category gaps are color (+8.26), object replacement
(+7.22), and deletion (+6.35). MasaCtrl is worst on color (+18.48) and deletion
(+16.23); the two valid style samples are also severe (+34.53) but are not a
stable category estimate. Pix2Pix-Zero is worst on deletion (+11.91), object
addition (+11.63), and pose (+11.21).

### 5. SSIM outside the edit mask

The evaluator blackens the edited region in both images and runs TorchMetrics
SSIM over the whole frame. Higher is better. This shares two artifacts with
LPIPS: the identical black region counts as perfectly preserved, and local
windows crossing the hard black-mask boundary are not a natural image
comparison. It is not an average of an SSIM map restricted to valid unedited
centres.

The paired losses are 0.34 for PnP, 1.52 for MasaCtrl, and 0.54 for
Pix2Pix-Zero; P2P improves by 4.83. SSIM and LPIPS gap correlations are 0.55 for
PnP, 0.83 for MasaCtrl, and 0.79 for Pix2Pix-Zero, consistent with shared
sensitivity to contour/texture destruction. MasaCtrl's broad SSIM loss and
Pix2Pix-Zero's severe individual collapses remain visible despite the
score-inflating black region.

## Source/reconstruction branch versus edited target branch

I separately extracted the third saved panel and measured full-image MSE
against the source. This uses all 700 images and is not the official metric.
The rightmost edited-panel full-image MSE is shown alongside it.

| Editor | Source branch DirectInv | Source branch LoRA+DI | Source gap | Edited-output gap | Correlation of per-image gaps |
|---|---:|---:|---:|---:|---:|
| P2P | 154.93 | 28.19 | -126.75 | -70.35 | 0.90 |
| PnP | 28.239 | 28.241 | +0.002 | +5.52 | -0.03 |
| MasaCtrl | 28.19 | 28.42 | +0.24 | +31.18 | 0.03 |
| Pix2Pix-Zero | 27.956 | 27.957 | +0.001 | +8.78 | 0.05 |

Approximately 28 MSE units is the common VAE encode/decode floor here.
MasaCtrl and Pix2Pix-Zero reconstruct their source branch nearly perfectly, but
their edited output becomes worse. Their per-image source gap has almost zero
correlation with the edited-output gap. P2P's LoRA source branch improves to the
VAE floor, and that improvement propagates strongly into its edit.

The PnP third panel is only the decoded near-clean `inverted_x[1]`, not a full
source-prompt denoising round trip, so its value must not be called a complete
PnP reconstruction test. It nevertheless shows that the observed target loss
does not correlate with this near-clean source representation. A dedicated
PnP source-prompt round trip is still required.

## Why the editors respond differently

### P2P

Both paths invert with the source prompt. DirectInv offsets are applied to the
source branch, and P2P attention control transfers source structure into the
target branch. The LoRA trajectory moves the source reconstruction from a poor
SD1.5 control result to the VAE floor, and the edited result improves with it.
This is the one editor for which source-branch and edited-branch gap correlation
is high (0.90).

Its relatively rare failures are mostly large, successful-looking edits on
artificial graphics (adding clouds, deleting a bus, extinguishing a fire) that
also redraw unmasked content. These do not overturn the approximately 90%
per-image win rate.

### PnP

PnP injects features from every stored inversion-trajectory latent into the
target denoising branch: convolutional features for about 80% of steps and
query/key attention features for about 50%. Its `DirectInv` integration does
not compute the same explicit offset list used by P2P/MasaCtrl; the full
trajectory is the reference signal.

The LoRA is intentionally trained to produce a trajectory different from
ordinary DDIM. Even when its endpoint or near-clean decode is accurate, its
intermediate base-UNet features can differ. PnP directly exposes those
differences to the editor. The small loss spread over contours far outside the
mask, together with the near-zero source-panel correlation, fits this mechanism
well.

### MasaCtrl

MasaCtrl applies DirectInv offsets only to the source branch and transfers
mutual self-attention from that branch into the target. The corrected source
latent can reconstruct at the VAE floor while its internal intermediate
features still differ enough to steer the target branch toward a different
identity, pose, or rendering.

There is also a concrete conditioning mismatch in the current fused code:

- classic DirectInv sets prompts to `[empty, target]`; the DDIM trajectory is
  therefore generated under the empty prompt;
- fused LoRA+DirectInv initializes inversion with `[source]`;
- it then switches back to `[empty, target]` to compute offsets and edit.

This simultaneously changes both the inversion model and inversion prompt.
It is the strongest code-level candidate for MasaCtrl's uniquely large loss.
The source output stays correct because offsets force the source latent back to
the trajectory, but the attention features copied into the uncorrected target
branch need not be compatible with the classic empty-conditioned trajectory.

Artificial graphics have a larger pixel-normalized MasaCtrl gap (+37.35) than
photographs (+21.48), so prompt-distribution expansion may help this editor.
However, PnP loses similarly on both groups, and Pix2Pix-Zero loses more on
photographs. Training data composition is therefore not a complete explanation
of the cross-editor pattern.

### Pix2Pix-Zero

Pix2Pix-Zero first denoises the inversion latent to build reference
cross-attention maps and DirectInv offsets. The edit pass then optimizes its
latent to match those maps after applying the edit direction, and applies the
same offsets step by step. A changed LoRA trajectory changes the offset path,
the reference attention maps, and the gradient-optimization landscape, even
when the source reconstruction at the end is numerically identical.

This is a tail-instability problem more than a uniform degradation. The median
official MSE gap is only +0.87, but the mean is +6.84; the top 10% of positive
cases contribute 66% of the positive MSE gap. The contact sheet shows complete
composition/style changes in the worst examples, not a subtle reconstruction
shift.

## Ranked explanation of the observed gap

1. **Editor-facing intermediate trajectory mismatch.** This is supported by
   almost perfect source branches but worse target branches for MasaCtrl and
   Pix2Pix-Zero, zero per-image correlation between the two gaps, and the way
   PnP/MasaCtrl/Pix2Pix-Zero explicitly consume intermediate latent/features.
2. **MasaCtrl prompt-conditioning confound.** This is a specific, testable
   implementation difference and likely explains why MasaCtrl is much worse
   than PnP.
3. **Contour/identity redraw far outside the intended region.** Spatial and
   visual diagnostics show central-subject geometry and edge losses, not only
   mask leakage at the boundary.
4. **Nonlinear tail failures, especially Pix2Pix-Zero.** A few complete
   regeneration failures dominate its means.
5. **Metric artifacts.** They inflate/dilute absolute values and omit many
   global edits, but corrected MSE preserves and strengthens the principal
   conclusions.
6. **Training-domain mismatch.** It is plausible for MasaCtrl/artificial
   graphics but cannot explain P2P's large improvement or Pix2Pix-Zero's larger
   photographic loss.

The standard uniform and cycle-loss rank-16 comparisons are effectively the
same. The completed lambda sweep is also almost flat for PnP and MasaCtrl.
Simply increasing the cycle-loss coefficient is therefore unlikely to solve
this failure mode.

## Next experiments, in order

1. **No-edit editor control:** set target prompt equal to source prompt and run
   every editor with DirectInv and LoRA+DirectInv. Score full-image output. This
   separates reference-trajectory incompatibility from deliberate edit
   strength.
2. **MasaCtrl three-way conditioning control:** compare (a) base inversion with
   empty prompt, (b) base inversion with source prompt, and (c) LoRA inversion
   with source prompt. Configuration (b) versus (a) measures the prompt
   confound; (c) versus (b) isolates LoRA.
3. **PnP true source round trip:** denoise each baseline and LoRA trajectory
   under the source prompt without target editing. The currently saved third
   PnP panel is not this experiment.
4. **Per-timestep compatibility diagnostic:** record latent-trajectory
   distance, DirectInv offset norm, base-UNet noise residual, and the actual
   editor-injected feature distance at every step. Correlate those values with
   per-image Structure/LPIPS/MSE gaps. This will identify whether early,
   middle, or late trajectory divergence causes each editor's loss.
5. **Pix2Pix-Zero crossing ablation:** independently swap its initial noisy
   endpoint, reference latent list/attention maps, and offset list between
   baseline and LoRA runs on the worst 20 examples. This will identify the
   unstable component without retraining.
6. **Report corrected metrics beside official metrics:** area-normalized
   unedited MSE/PSNR; SSIM-map average only on eroded unedited pixels; masked
   feature LPIPS that excludes boundary-contaminated activations; DINO
   Structure with inputs divided by 255. Keep official numbers for paper
   comparability.

Only after the per-timestep diagnostic should the training objective be
changed. A useful editor-independent target would minimize the base denoiser's
required per-step correction or multi-step round-trip drift, because that
directly expresses the intended property: a LoRA-generated inverse trajectory
that remains compatible with the unchanged base-model denoising process.

## Reproducibility artifacts

- `diff_inversion/eval/analyze_reconstruction_pixels.py`: area-corrected and
  spatial pixel analysis.
- `reports/reconstruction_metric_analysis/paired_metric_analysis.json`: paired
  official-metric summaries, bootstrap intervals, categories, and outliers.
- `reports/reconstruction_metric_analysis/pixel_spatial_analysis.json`: mask,
  distance, texture, source-group, category, and spatial-grid diagnostics.
- `reports/reconstruction_metric_analysis/source_panel_analysis.json`: separate
  third-panel and edited-panel experiment.
- `reports/reconstruction_metric_analysis/reconstruction_outliers_*.jpg`:
  source/mask/DirectInv/LoRA+DirectInv/error contact sheets.

