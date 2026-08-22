# Artificial-graphics trajectory mixture experiment

## Objective

Test whether the inversion LoRA reconstructs non-photographic images poorly because its
training trajectories are generated from a predominantly photographic prompt distribution.
Train a controlled LoRA variant on a mixture of the existing ReCap-COCO prompts and prompts
that explicitly request illustrations, paintings, cartoons, renders, posters, diagrams, and
other artificial graphics.

The first experiment is a data-distribution experiment. Do not change LoRA architecture,
loss, precision, trajectory length, guidance, optimizer, scheduler, batch size, or training
budget at the same time.

## Important fact about the current pipeline

`diff_inversion/data/generate_sdxl_samples.py` does not construct trajectories from dataset
images. It reads a prompt JSONL file, samples a new image with Stable Diffusion, and saves the
sampling trajectory. ReCap-COCO currently contributes prompts, not COCO pixels.

Consequently, adding an image-classification dataset such as DomainNet would not help unless
we first captioned its images or implemented a different image-inversion trajectory pipeline.
For the initial experiment, use a prompt dataset whose language explicitly covers artificial
visual styles.

## Chosen dataset: DiffusionDB prompt metadata

Use the official `poloclub/diffusiondb` DiffusionDB 2M metadata table:

- Dataset: `https://huggingface.co/datasets/poloclub/diffusiondb`
- Metadata file: `metadata.parquet`
- Do not download the image ZIP files.
- Relevant columns: `prompt`, `prompt_nsfw`, `image_nsfw`, `width`, and `height`.
- DiffusionDB is distributed under CC0 1.0 according to its official dataset card.

Why this is the primary choice:

1. The existing pipeline needs prompts, and DiffusionDB exposes prompt-only metadata.
2. Its prompts were written by text-to-image users and contain explicit style language.
3. The metadata is public and does not require downloading the approximately 1.6 TB image
   subset.
4. It is operationally portable to another HPC and does not require a captioning model.

JourneyDB is a possible second experiment because it has rich Midjourney style prompts, but
it is gated, uses customized terms, and its full release is several terabytes. Do not make it
the first implementation. DomainNet is a later option only for a pipeline that actually
inverts source images or captions them first.

## Prompt subset construction

Create exactly 7,610 artificial-style prompts. Together with the existing 30,441
ReCap-COCO prompts, this corresponds to an approximately 80/20 source mixture by unique
prompt count.

Use seed 42 for every random selection. Preserve the selected manifest in the repository or
as a checksum-addressed data artifact so another HPC selects the same prompts.

### Initial filtering

Apply these filters before style classification:

- Require a non-empty string with at least four whitespace-separated words.
- Deduplicate by Unicode-normalizing, lowercasing, collapsing whitespace, and stripping
  leading/trailing punctuation. Keep the original prompt text as the generation prompt.
- Reject prompts longer than 75 tokens under the SD1.5 CLIP tokenizer. Do not truncate; a
  truncated style phrase could silently move a prompt into the wrong domain.
- Reject `prompt_nsfw >= 0.1` or `image_nsfw >= 0.1` and reject rows with missing NSFW scores.
- Reject prompts dominated by non-Latin text. Record the exact language/character heuristic.
- Reject obvious empty boilerplate, URLs, Discord mentions, and repeated-token spam.
- Reject prompts containing photographic-target terms such as `photograph`, `photography`,
  `photorealistic`, `DSLR`, `35mm`, `bokeh`, or explicit camera/lens specifications. The goal
  is non-photographic coverage, not generic synthetic photographs.

### Style strata

Build five mutually exclusive strata and sample 1,522 prompts from each:

1. `illustration_vector`: illustration, vector art, clip art, flat design, iconography.
2. `cartoon_comic_anime`: cartoon, comic, manga, anime, cel-shaded artwork.
3. `painting_digital_art`: painting, watercolor, oil painting, gouache, digital art,
   concept art.
4. `render_3d`: 3D render, CGI, Blender, Octane render, isometric render, low-poly art.
5. `graphic_design`: poster, logo, graphic design, infographic, diagram, UI design,
   typography, pixel art.

Implement deterministic precedence in the order above when a prompt matches multiple
strata, and save both all matched labels and the selected primary label. Keep the keyword
lists in configuration rather than embedding them only in Python.

The preparation output must be a JSONL file compatible with the current generator. Each
record should contain at least:

```json
{
  "prompt": "...",
  "source_dataset": "diffusiondb",
  "style_stratum": "graphic_design",
  "source_row_id": 123,
  "selection_seed": 42
}
```

Also write a summary JSON containing input row count, rejection counts by reason, duplicate
count, eligible count per stratum, selected count per stratum, and a SHA-256 checksum of the
final JSONL.

Manually inspect a fixed random sample of 50 prompts per stratum before generating expensive
trajectories. This inspection is for prompt quality and correct categorization only; do not
select prompts based on PIE-Bench results.

## Trajectory generation

Generate the selected artificial-prompt trajectories separately from the ReCap trajectories:

```text
data/processed/sd15_trajectories_diffusiondb_artificial_fp32/all/
```

Keep all artificial training trajectories in this single `all/` directory. Do not create
train/validation/test trajectory directories.

Match the existing full-FP32 ReCap trajectory setup exactly:

- Stable Diffusion 1.5.
- Model dtype: float32; model variant: null.
- Saved trajectories, conditioning, and targets: float32.
- DDIM sampling steps: 50.
- Sampling/generation CFG: 1.0 for the first experiment.
- Empty negative prompt.
- Resolution: 512 by 512.
- One prompt/image per generation trajectory.
- Save `final.png`, `prompt.json`, `meta.json`, `timesteps.json`,
  `latents/trajectory.pt`, `conditioning.pt`, and `targets/target_eps.pt`.
- Preserve `source_dataset` and `style_stratum` in sample metadata.
- Use deterministic seeds derived from the global prompt index plus the configured base seed.
- Generate or recompute unconditional targets only if the selected training-target mode needs
  them; do not mix target-cache formats inside one trajectory root.

Use repository-relative paths and data/checkpoint symlinks. Do not put absolute HPC paths in
Hydra configuration. All UV, Hugging Face, Torch, Triton, CUDA, W&B, Matplotlib, and temporary
caches must live under `$SLURM_TMPDIR` and be deleted at job exit. Only final trajectories,
checkpoints, images, metrics, and logs belong on persistent storage.

## Training mixture

Add both trajectory roots to the training dataset:

```yaml
data:
  root_dirs:
    - data/processed/sd15_trajectories_stacked_fp32/all
    - data/processed/sd15_trajectories_diffusiondb_artificial_fp32/all
```

Implement a source-aware weighted sampler rather than relying on directory concatenation.
Each transition draw should independently choose source mass as follows:

```yaml
sampling:
  mode: source_mixture
  source_weights:
    recap_coco: 0.80
    diffusiondb_artificial: 0.20
  num_samples: 1522050
  replacement: true
  seed: 42
```

`1,522,050` is the existing ReCap training budget: 30,441 trajectories times 50 transitions.
With batch size 128 this remains 11,892 optimizer updates because the final partial batch is
included. Keep `max_train_steps=11892`; do not increase it because the mixed pool is larger.
This ensures the comparison measures data composition rather than 25% more optimization.

The sampler must assign source probability mass per transition, not per directory size. A
normal batch is allowed to contain transitions from multiple source images and both sources.
Do not group a batch by image or trajectory. Preserve uniform sampling over the 50 trajectory
transitions within each source for this first experiment.

The source-mixture sampler and final-tail sampler are separate experiments. Do not enable
final-tail oversampling in the first source-mixture run. If they are later composed, specify a
joint distribution explicitly instead of multiplying weights without tests.

## First training run

Run rank 16 first. Use the exact ReCap-only baseline configuration except for the source-aware
sampler and second trajectory root:

- LoRA: r16 with the repository's existing alpha/dropout/target-module settings.
- Precision: full float32 model and training tensors.
- Batch size: 128.
- Gradient accumulation: 1.
- Optimizer update after every batch.
- Optimizer, learning rate, weight decay, clipping, and scheduler: unchanged from baseline.
- Training target/loss: unchanged from the baseline being compared.
- Total optimizer updates: 11,892.
- W&B enabled if credentials are available through `.env`; never store a key in config.
- Save normal periodic training-state checkpoints and a final adapter/training state.

Do not launch r8 and r32 until the r16 result supports the hypothesis. If r16 improves the
artificial subset without an unacceptable natural-image regression, repeat r8 and r32 with
the identical data manifest and training budget.

## Controls and evaluation

The primary comparison is:

```text
ReCap-only r16, uniform steps, fixed 11,892 updates
versus
80% ReCap + 20% DiffusionDB-artificial r16, uniform steps, fixed 11,892 updates
```

Do not compare against a baseline with a different rank, number of updates, CFG, precision,
loss, or tail-sampling policy.

Run the full PIE-Bench pipeline for every editor used elsewhere in this project, both pure
LoRA inversion and LoRA + Direct Inversion. Do not evaluate only PnP.

Before viewing method results, create and freeze a PIE-Bench image manifest with two labels:

- `artificial_graphic`: illustrations, cartoons, paintings, renders, posters, diagrams,
  vector-like graphics, and similar non-photographic images.
- `photographic`: photographs or photorealistic scenes.

If an image cannot be assigned confidently, mark it `uncertain` and report it separately;
do not silently force it into the photographic group. Store image IDs and labels in JSONL.
The labeling process must not inspect outputs from either model.

Report reconstruction and editing metrics for:

1. all PIE-Bench images;
2. the frozen artificial-graphic subset;
3. the frozen photographic subset;
4. each DiffusionDB style stratum on a small held-out prompt-generated diagnostic set.

Use per-image paired comparisons and report mean, standard deviation, and bootstrap 95%
confidence intervals for the difference between baseline and mixture. For every metric,
state explicitly whether higher or lower is better.

Treat the experiment as promising if artificial-graphic reconstruction improves consistently
across structure/perceptual metrics while photographic reconstruction and overall editing
quality do not materially regress. Use a provisional photographic non-inferiority tolerance
of 2% relative; report the raw values even when the tolerance is met.

## Required implementation work

1. Add a Hydra data config for DiffusionDB metadata and prepared prompt JSONL paths.
2. Add a deterministic prompt-preparation command implementing the filters, strata, summary,
   and checksum above.
3. Add an SD1.5 trajectory-gather config with its own artificial trajectory output root.
4. Preserve source and stratum metadata in generated sample `meta.json` files.
5. Extend `LatentTrajectoryDataset` metadata so each transition exposes its source root/source
   name without loading tensors.
6. Add `sampling=source_mixture` and a deterministic weighted sampler with configurable source
   weights and draw count.
7. Keep validation uniformly sampled and separate from the weighted training sampler.
8. Add unit tests for deterministic selection, disjoint strata, exact selected counts,
   source-weight normalization, draw count, and transition-to-source mapping.
9. Add a short CPU sampler smoke test and a small GPU trajectory/training smoke test before
   submitting the full generation or training jobs.
10. Add Slurm launchers using the target HPC's modules/partitions but repository-relative Hydra
    paths and `$SLURM_TMPDIR` runtime caches.

Do not download DiffusionDB images, launch full jobs, or alter existing baseline artifacts
until the prepared prompt statistics and 250-prompt manual quality sample have been reviewed.

## Follow-up ablations

Only after the 80/20 r16 result:

- Test 90/10 and 60/40 mixtures with the same 11,892-update budget.
- Test source mixture plus final-tail sampling as a separately named experiment.
- Consider JourneyDB prompts if DiffusionDB lacks sufficient high-quality poster,
  infographic, or illustration coverage and its access terms are accepted.
- Consider DomainNet only when implementing trajectories based on actual source-image
  inversion or a separately validated image-captioning stage.
