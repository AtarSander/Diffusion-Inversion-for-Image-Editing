# PIE-Bench best/worst reconstruction examples

Use this procedure to select the 25 PIE-Bench examples that most improve and
most degrade reconstruction metrics for each editor.  It is for comparing
failure cases between training approaches; it does not rank edit quality.

## Ranking

`scripts/analysis/select_pie_reconstruction_examples.py` reads per-image metric
CSVs and ranks Pix2Pix-Zero, MasaCtrl, and PnP independently.  The recommended
mode is `mean_degradation` with `--require-all-metrics`:

- Metrics: structure distance, unedited-region LPIPS, unedited-region MSE,
  unedited-region PSNR, and unedited-region SSIM.
- CLIP/edit metrics are intentionally excluded.
- Each metric is standardized across valid images for that editor and run.
- The bad-direction z-scores are averaged with equal weights: high structure
  distance/LPIPS/MSE and low PSNR/SSIM are worse.
- The lowest 25 composite values are `best`; the highest 25 are `worst`.

This identifies images that contribute most to the aggregate reconstruction
results.  `--require-all-metrics` avoids changing the composite because an
image mask leaves no unedited region, in which case unedited-region metrics are
undefined.

## Run

`ROOT` is an evaluation root containing `metrics/<method>.csv` and generated
editor panels.  The current selector expects methods named
`lora+directinversion+pix2pix-zero`, `lora+directinversion+masactrl`, and
`lora+directinversion+pnp`; adapt its `method` assignment if the other run uses
different names.

```bash
python3 scripts/analysis/select_pie_reconstruction_examples.py \
  --root "$ROOT" \
  --data data/raw/PIE-Bench_v1 \
  --output "$SELECTION_ROOT" \
  --count 25 \
  --score-mode mean_degradation \
  --require-all-metrics

python3 scripts/analysis/make_pie_reconstruction_contact_sheets.py \
  --selection-root "$SELECTION_ROOT" \
  --output "$SELECTION_ROOT/contact_sheets"
```

The selector writes `README.json` (methodology and ranked IDs), plus
`<editor>/best.csv`, `<editor>/worst.csv`, JSON manifests, and linked/copied
editor panels.  The renderer creates one labeled 5x5 JPEG for each editor and
best/worst set.

## Compare two approaches

Run the exact command on both evaluation roots.  Compare the `best_ids` and
`worst_ids` in their `README.json` files per editor.  Report top-25 and
bottom-25 ID overlap; for a stronger comparison, join all shared image IDs from
the CSVs and calculate rank correlation of the composite scores.  The scores
are standardized within each run, so compare ranks/IDs rather than raw composite
values across runs.
