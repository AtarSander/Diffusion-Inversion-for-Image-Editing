# Multi-seed OVG transfer benchmarks

Both tasks compare upstream DDIM, DDIM with our timestep convention, inversion-only
r16 LoRA, and OVG on the original text-conditioned SD1.5 model, in FP32 with 50
steps and CFG 1. These are transfer experiments, not reproductions of the paper's
task-trained models. The adapter is `sd15_inversion_lora_r16_uniform_e1`.

## Fixed evaluation images, four stochastic seeds

Use seeds 42, 43, 44 and 45. `sample_seed=42` fixes image selection and ordering;
`seed` controls stochastic model inputs (including sampled VAE latents). Within
each run the same VAE encoding is reused across inversion methods. BBBC021 uses
576 of 3,848 labelled fields at x16, retaining the completed seed-42 evaluation.
The additional runs do not select fresh images or increase the evaluated pool.

Edges2Shoes uses the 200 official validation pairs from the pix2pix release:
https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz .
The archive is staged in node-local storage; only `val/` files are retained,
unchanged, with checksums in `data/processed/edges2shoes/val.csv`. The
training pairs are not extracted. Upstream splits each 512x256 pair into the
left edge image and right shoe reference. Evaluation augmentation is disabled.
These 200 pairs are much less than 15% of the full dataset. They are not claimed
to be the paper's unpublished exact evaluation selection.

The Edges2Shoes config inherits common model settings from the BBBC021 config,
replaces source/target prompts and task/data paths, and uses upstream Edges2Shoes
OVG coefficients. No category annotations are supplied with these pairs, so
placeholder class indices are not used to report kNN recovery. All reference
metrics compare generated shoes with their paired shoe photographs. This is
edge-to-photo translation, not a downsampled superresolution task.

## Launching

```bash
sbatch scripts/slurm/prepare_edges2shoes.sbatch
# Prefer h32 / gpu-m, or override BOTH partition and node for h86 / gpu-l.
sbatch --nodelist=h32 --cpus-per-task=4 --mem=40G \
  scripts/slurm/benchmark_bbbc021_h32.sbatch seed=43
sbatch --nodelist=h32 --cpus-per-task=4 --mem=40G \
  scripts/slurm/benchmark_bbbc021_h32.sbatch \
  --config-name=eval/edges2shoes seed=42
```

Each GPU job computes metrics and verifies finite outputs, full sample coverage,
and non-identical LoRA/matched-DDIM inversion endpoints before success. Use
independent jobs, not arrays. After all seeds succeed, submit
`scripts/slurm/summarize_ovg_seeds.sbatch` with an `afterok` dependency on their
job IDs, `--output-dir`, and the four result directories. This CPU job verifies
model/protocol agreement and identical image selection, then writes
`mean_metrics.json` and `mean_metrics.csv`. Each metric is averaged over the
four per-seed values; FID is averaged, not recalculated on pooled images.
Invalid kNN recovery measurements are omitted rather than averaged as successes.

No full datasets are evaluated and no models are retrained. All runtime caches
and environments stay node-local and are deleted at job exit.
