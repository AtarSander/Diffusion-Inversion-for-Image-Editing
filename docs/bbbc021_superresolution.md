# BBBC021 superresolution with inversion LoRA

The integration imports dataset handling, degradation, benchmark orchestration,
sampling, HDF5 export and metrics directly from
[nicoboou/ovg](https://github.com/nicoboou/ovg), pinned to
`e13fe392ba8f0738ce2f041ee356bb2b5a0171db`. Run `bash scripts/setup_ovg.sh`
after cloning this project. The external checkout stays unmodified and is not
vendored or installed with its incompatible dependency pins.

## What this experiment measures

`config/eval/bbbc021.yaml` evaluates transfer of our existing SD1.5 adapters to
microscopy superresolution. It uses OVG's 256x256 images, nearest-neighbor
downsample/upsample by 16, 50 steps and guidance 1, with identical source images
and sampled VAE encodings across methods. Change `data.scale_factor` to 8 or 32
for the other degradation levels. FP32 is selected with
`benchmark.mixed_precision=no`.

Methods are upstream DDIM, DDIM using our inversion timestep convention,
our LoRA inversion, and upstream OVG. The matched DDIM control and LoRA use the
same `DDIMInverseScheduler`; this matters because upstream DDIM evaluates at
`max(next_t - step, 0)` while our trained adapter evaluates at `next_t`.
All sampling uses the base UNet with the inversion adapter disabled, including
after an inversion exception. Adapter key mismatches fail the run.

The paper instead trains a microscopy model with low/high-resolution class
conditioning on disjoint unpaired training images. Our existing adapters were
trained against the original text-conditioned SD1.5 model. Results from this
transfer experiment must be compared against its own controls, not claimed as
an exact reproduction of the paper's table. Source/target microscopy prompts
are explicit configurable experimental choices, not author-provided prompts.

## Required data

The public OVG repository does **not** provide the BBBC021 preparation/download
script, CSV splits, channel normalization recipe or pretrained microscopy
checkpoint. Its test config contains `CSV_PATH` and `CHECKPOINT_PATH.pt`.
Obtain the author's prepared RGB images and evaluation CSV for the closest
comparison. For our transfer experiment, automatic preparation is available:

```bash
sbatch scripts/slurm/prepare_bbbc021.sbatch
```

`config/data/bbbc021_prepare.yaml` specifies the local protocol. The script
downloads the official metadata and all 55 plate archives (about 45 GB compressed).
Each archive is staged in node-local temporary storage, processed, and removed;
only prepared RGB PNGs, metadata, manifests and checksums persist through the
repository's `data` symlink. It does not extract 39,600 TIFF files on shared storage.
Completed plates are checksummed and skipped on restart. `summary.json` is written
only after all 13,200 fields are prepared successfully.

The explicitly **local**, non-author-provided conversion maps Actin to red,
Tubulin to green and DAPI to blue. Each field/channel is independently clipped
at its 0.1/99.9 intensity percentiles, linearly mapped to uint8, then the composite
is bilinearly resized to 256x256 without cropping. This intensity normalization
changes inter-field brightness relationships and must remain fixed across all
methods. Degenerate channels cause a failure, not silent sample substitution.
LR images are constructed later by the upstream nearest-neighbor degradation.

`all.csv` lists all fields. `test.csv` contains only fields with an official
compound-and-concentration MOA annotation; unknown fields have class -1 in
`all.csv` and are not assigned a fabricated biological class. Class indices
follow alphabetical MOA order in `classes.json`. This is an evaluation-only pool
for our existing natural-image-trained adapters, **not** the paper's unpublished
train/test split. Do not train on it or treat the generic upstream kNN diagnostic
as the official leave-one-compound-out MOA classification benchmark. Freeze a
separate held-out split before any future microscopy training.

Place prepared images under `data/raw/BBBC021` and the CSV at
`data/processed/bbbc021/test.csv`, or override those paths in Hydra. CSV schema:

```csv
path,class_idx,res_class
images/example_001.png,0,1
images/example_002.png,1,1
```

`path` is relative to `data.image_root`; `class_idx` is the biological category
used for the upstream kNN metric; `res_class=1` marks the HR reference from which
paired test LR inputs are constructed. These are schema examples, not dataset
labels. The runner checks image readability, RGB mode and duplicates before
evaluation; it does not allow upstream's silent replacement of unreadable
images to bias the sample set. Evaluation pairing does not imply paired
training. Never train adapters on the evaluation split.

## Running

```bash
bash scripts/setup_ovg.sh
mkdir -p logs/slurm
# Tiny network, real schedulers/PEFT; no dataset or pretrained downloads needed.
sbatch --time=00:20:00 scripts/slurm/benchmark_bbbc021_h32.sbatch --smoke
# Once prepared data are available:
sbatch scripts/slurm/benchmark_bbbc021_h32.sbatch
# Cycle loss, for example:
sbatch scripts/slurm/benchmark_bbbc021_h32.sbatch \
  adapter_checkpoint=checkpoints/sd15_inversion_lora_r16_conditional_cycle_cfg1_bs120_e1/checkpoint_final.pt \
  output_dir=reports/eval/bbbc021/cycle_r16_x16_seed42
```

For r8/r32 override both `lora=r8` / `lora=r32` and `adapter_checkpoint`.
The upstream public test config defaults to 32 images. Our configuration uses
576 images (14.97% of the 3,848 officially labelled fields; 4.36% of all fields),
selected with seed 42 and reused for every method. The exact selection is saved
in `sample_manifest.csv`. This does not establish the paper's evaluation size.
Set `benchmark.num_samples` to change it and run three independent seeds
(e.g. 42, 43, 44, a local choice)
before reporting means. Freeze the evaluation manifest and protocol first.
Launchers use independent jobs rather than Slurm arrays. Runtime caches and the
environment are node-local and removed on exit; images/metrics/logs persist.

## Outputs and upstream limitations

Each run writes `summary.json`, resolved config, `sample_manifest.csv`,
`resolved_test.csv`, visualization tensors, and compressed HDF5 files containing
HR references, degraded inputs, generated outputs and inverted endpoints.
Reported upstream metrics include PSNR, MS-SSIM, LPIPS, HaarPSI, structure
distance, FID and class kNN recovery. Metrics compare generated HR against HR.
The bridge preserves upstream metric normalization for reproducibility.

The released runner includes spectral plots but does not implement the paper's
reported HF/LF/decorrelation score suite. These columns are not claimed or
fabricated. DirectInv, Null-Class and ReNoise are also absent from the released
method registry. Its custom-checkpoint loader is incorrectly nested in a
function, so it cannot currently be called as a CheckpointManager method.

To reproduce the paper backbone experiment, obtain its prepared split and
microscopy denoiser/class-embedding/VAE checkpoint, fix that loader, generate
teacher trajectories under its low-resolution conditioning, and train a new
inversion LoRA against that frozen model. Reusing our natural-image SD1.5 LoRA
weights on a different trained teacher would not test the same learned inverse.
The preparation job downloads data only; it does not launch training or evaluation.
