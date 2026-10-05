# SD1.4 LoRA inversion retraining handoff

## Purpose

This file is the execution brief for an agent running the experiments on a second HPC.

The goal is to repeat the relevant SD1.5 LoRA-inversion experiments with Stable Diffusion 1.4 so that P2P, MasaCtrl, and Pix2Pix-Zero can be compared directly with the original Direct Inversion paper results. Do not rerun PnP as part of this SD1.4 matrix: Table 1 of the paper marks PnP as the SD1.5 exception and states that the other three editors use SD1.4.

Paper source: [Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code](https://arxiv.org/html/2310.01506v2). Table 1 identifies the model exception, and Appendix D specifies SD1.4, 50 diffusion steps, and inversion guidance 1 for the main protocol.

This is a controlled model-version experiment. Apart from changing SD1.5 to SD1.4, keep the established trajectory, training, inversion, editing, and metric protocols fixed unless this document explicitly says otherwise.

## Scope

The required editors are:

1. Prompt-to-Prompt (P2P)
2. MasaCtrl
3. Pix2Pix-Zero

For every trained checkpoint, evaluate both inversion variants:

- pure LoRA inversion plus the editor, with no Direct Inversion code;
- LoRA inversion plus Direct Inversion part II plus the editor.

One LoRA checkpoint is shared across all three editors. Do not train editor-specific LoRAs, and do not train a second adapter merely because Direct Inversion is enabled during evaluation.

Only already verified SD1.5 training setups belong in the SD1.4 rerun. The conditional-cycle loss is experimental and remains an SD1.5-only experiment for now; do not port, train, or evaluate it on SD1.4.

## Source revision and readiness gate

Work from the img_edition branch containing all of the following:

- corrected pure-LoRA editor paths that do not invoke Direct Inversion;
- corrected LoRA plus Direct Inversion editor paths;
- explicit model-key propagation so a LoRA evaluation does not default back to SD1.5;
- verified standard conditional CFG1 training with uniform sampling;
- verified standard conditional CFG1 training with final-tail sampling.

Record the exact Git commit used for every trajectory, checkpoint, image set, and metric summary. If the required changes are not yet committed on the remote branch, stop and ask for the updated commit rather than reconstructing the feature from this document.

## Fixed ReCap-COCO prompt manifest

Use the existing prompt-generation pipeline. It creates a new Stable Diffusion image and sampling trajectory from each ReCap-COCO prompt; it does not invert the original COCO image.

The prompt manifest must be:

    data/processed/recap_coco/all_prompts.jsonl

Expected properties:

- records: 30,441;
- SHA-256: fee51b91c8b226d85c523c398b2633509c005d107027298bf4114d8ca68d4d0a;
- one generated image and one trajectory per JSONL record.

Before generation, run:

    wc -l data/processed/recap_coco/all_prompts.jsonl
    sha256sum data/processed/recap_coco/all_prompts.jsonl

Do not proceed if either value differs. Copy or regenerate the manifest deterministically before consuming GPU time.

## SD1.4 model and trajectory protocol

Add a Hydra model config named model/sd14.yaml. It should use the existing Stable Diffusion pipeline implementation with:

- model ID: CompVis/stable-diffusion-v1-4;
- scheduler: DDIM;
- model dtype: float32;
- variant: null;
- revision: null unless the local model installation demonstrably requires a specific revision;
- 50 sampling steps;
- guidance scale: 1.0;
- resolution: 512 by 512;
- CUDA required for full generation.

Add a dedicated gather config and never reuse the SD1.5 output directory.

Required output root:

    data/processed/sd14_trajectories_stacked_fp32/all

Each sample must contain:

- final.png;
- prompt.json;
- meta.json;
- timesteps.json;
- latents/trajectory.pt;
- conditioning.pt;
- targets/target_eps.pt.

Generation rules:

- full FP32 model execution;
- latents, conditioning, and target epsilon saved as float32;
- empty negative prompt;
- one prompt per generation call;
- deterministic seed derived from the global prompt index and the configured base seed;
- 30,441 distinct sample directories in one all directory;
- no train, validation, or test trajectory subdirectories;
- no overwriting of SD1.5 trajectories;
- non-overlapping start-index/count shards if generation is split across jobs.

The existing nine-way SD1.5 ranges total 30,441 and may be reused as explicit ordinary jobs:

| Shard | Start | Count |
|---:|---:|---:|
| 0 | 0 | 3425 |
| 1 | 3425 | 3425 |
| 2 | 6850 | 3425 |
| 3 | 10275 | 3425 |
| 4 | 13700 | 3425 |
| 5 | 17125 | 3425 |
| 6 | 20550 | 3425 |
| 7 | 23975 | 3425 |
| 8 | 27400 | 3041 |

Prefer ordinary jobs over a rate-limited Slurm array. Every shard may write to the same all directory because the index ranges are disjoint.

After generation, validate all of the following before training:

- exactly 30,441 sample directories;
- no missing indices from sample_000000 through sample_030440;
- no unexpected indices or duplicate paths;
- 51 stored latents and 50 target-epsilon transitions per sample;
- all required files present;
- all stored floating tensors are float32;
- the active model ID in run_config.json and sample metadata is SD1.4;
- no SD1.5 path or model identifier occurs in the SD1.4 root;
- several random DDIM transitions and cached targets pass the existing cache smoke checks.

Save a machine-readable validation summary and a list of missing or corrupt samples. Repair only the failed indices.

## Training configuration held fixed

Create model, data, gather, training, and evaluation config nodes for SD1.4. Do not create separate config files merely to express FP32; precision should be a model override or a small model-config field.

Use the same full-FP32 baseline training protocol as SD1.5:

- 30,441 trajectories;
- 50 transitions per trajectory;
- 1,522,050 transitions per epoch;
- batch size 128;
- gradient accumulation 1;
- optimizer update after every batch;
- one epoch;
- max_train_steps 11,892, including the final partial batch;
- AdamW;
- learning rate 5e-5;
- weight decay 0;
- maximum gradient norm 1.0;
- cosine scheduler;
- 238 warmup steps, which is 2 percent of 11,892 rounded to the existing value;
- seed 42 for training and samplers;
- LoRA targets to_q, to_k, to_v, and to_out.0;
- LoRA dropout 0;
- existing rank-specific alpha values: r8/alpha4, r16/alpha8, r32/alpha16;
- normal periodic adapter and training-state checkpoints;
- a checkpoint near the wall-time limit and a final checkpoint;
- W&B only through environment credentials; never put a key in Hydra or a launcher.

For uniform sampling, shuffle individual transitions normally. A batch may contain transitions from multiple images.

For final-tail sampling, preserve the existing policy exactly:

- final_step_fraction: 0.10;
- target_draw_fraction: 0.50;
- replacement: true;
- number of draws: 1,522,050;
- 11,892 optimizer updates.


## Prioritized training matrix

The SD1.5 winners by lowest structure distance were:

| Editor | Best uniform CFG1 | Best final-tail CFG1 |
|---|---|---|
| P2P | r8 | r8 |
| MasaCtrl | r16 | r32 |
| Pix2Pix-Zero | r16 | r8 |

Because a checkpoint is shared across editors, the minimal old-method rank coverage is therefore:

- uniform CFG1: r8 and r16;
- final-tail CFG1: r8 and r32.

Run work in this order:

### Priority 1: proven standard CFG1 configurations

1. Standard conditional loss, uniform sampling, r8.
2. Standard conditional loss, uniform sampling, r16.
3. Standard conditional loss, final-tail sampling, r8.
4. Standard conditional loss, final-tail sampling, r32.

Do not initially spend time on uniform r32 or final-tail r16. They were not the SD1.5 winners for any of the three SD1.4 editors. Add them only after the core matrix is complete or if a result indicates a rank trend that requires them.


### Priority 2: lower-value verified ablations

The SD1.5 CFG2.5 standard runs were worse on structure for P2P, MasaCtrl, and Pix2Pix-Zero. If resources remain, generate a separate SD1.4 CFG2.5 trajectory root and train only uniform r8 first. Never mix CFG1 and CFG2.5 trajectory caches.

Do not rerun branch-pair CFG1 by default. Its corrected SD1.5 results were catastrophically worse for these three editors. It is not part of the required SD1.4 core matrix unless separately requested.

Do not include the DiffusionDB artificial-prompt mixture in this experiment. That is a different data-distribution experiment and would confound the model-version comparison.

## Required controls

Before interpreting LoRA results, regenerate local SD1.4 controls with the same checked-out code and evaluation environment:

- DDIM plus P2P;
- DirectInv plus P2P;
- DDIM plus MasaCtrl;
- DirectInv plus MasaCtrl;
- DDIM plus Pix2Pix-Zero;
- DirectInv plus Pix2Pix-Zero.

These controls should closely reproduce the paper rows before they become the primary comparison. If they do not, debug the editor and metric pipeline before blaming a LoRA checkpoint.

The paper reference means are available in reports/pnp_lora_one_epoch_results.tex and reports/direct_inversion_reproduction_comparison.tex.

## Editor evaluation protocol

Make the SD1.4 model key explicit in every editor invocation, including LoRA runs:

    CompVis/stable-diffusion-v1-4

This is critical because the current editor code historically selected SD1.5 whenever use_lora was true.

Hold these settings fixed across controls and LoRA rows:

- 50 DDIM steps;
- inversion guidance 1.0 for the CFG1 matrix;
- existing editor generation guidance 7.5;
- the existing per-editor attention and editing hyperparameters;
- the same PIE-Bench source images, prompts, masks, and 700-image manifest;
- the same inference dtype for baseline and LoRA comparisons, normally FP16 in the editor pipeline;
- the same metric code and mask handling.

For every selected checkpoint, generate all six image sets:

1. LoRA plus P2P;
2. LoRA plus DirectInv plus P2P;
3. LoRA plus MasaCtrl;
4. LoRA plus DirectInv plus MasaCtrl;
5. LoRA plus Pix2Pix-Zero;
6. LoRA plus DirectInv plus Pix2Pix-Zero.

Pure LoRA paths must not instantiate or call DirectInversion. Fused paths must use LoRA for Algorithm 1 part I inversion and preserve the existing Direct Inversion part II behavior.

Chain metric calculation immediately after each successful image-generation job when practical. A generation failure must prevent metrics from running for that image set.

Store outputs under a new SD1.4 namespace containing at least:

- model version;
- training loss family;
- sampling policy;
- rank;
- inversion variant;
- editor;
- checkpoint step or final marker.

Never write SD1.4 images or metrics into an existing SD1.5 directory.

## Metrics and reporting

Run the existing seven PIE-Bench metrics on all 700 images:

- Structure Distance: lower is better;
- PSNR: higher is better;
- LPIPS: lower is better;
- MSE: lower is better;
- SSIM: higher is better;
- whole-image CLIP similarity: higher is better;
- edited-region CLIP similarity: higher is better.

Use the existing PIE-Bench masks and the same empty-unedited-region policy used by prior project metrics. Report valid counts for mask-restricted metrics.

Save per-image values as well as aggregate mean and standard deviation. Do not save only a formatted table.

Create an SD1.4 comparison table with, for each editor:

1. paper original DDIM;
2. paper original DirectInv;
3. local SD1.4 DDIM reproduction;
4. local SD1.4 DirectInv reproduction;
5. selected standard SD1.4 LoRA rows;
6. selected standard SD1.4 LoRA plus DirectInv rows;

Bold the best result within each editor and metric. Underline experimental values that beat the paper original DirectInv result for the same editor and metric. Keep raw machine-readable metrics as the source of truth.

## Hydra structure to add

Use composition rather than copy-pasted monolithic configs. The expected additions are approximately:

- config/model/sd14.yaml;
- config/data/sd14_trajectories.yaml;
- config/sample_gather_sd14.yaml;
- config/train_sd14.yaml;
- config/train/sd14.yaml;
- SD1.4 evaluation overrides or a model node reusable by all three editors;
- SD1.4 baseline and metric output roots.

Reuse the existing independent nodes for:

- lora/r8, lora/r16, lora/r32;
- sampling/uniform and sampling/final_tail;
- training_target/conditional;
- wandb profiles;
- learning-rate scheduler.

All paths inside Hydra must be repository-relative and resolve through data/checkpoints symlinks. Do not put HPC-specific absolute paths into config files.

## Slurm and cache requirements on the second HPC

First inspect that HPC's tutorial, partitions, modules, GPU types, memory, wall-time, and local temporary-storage conventions. Do not copy h32/h86 node names to a different cluster.

Every launcher must:

- write stdout and stderr under logs/slurm or another dedicated log directory;
- request email for END, FAIL, and TIME_LIMIT if the cluster supports it;
- create the Python environment on the compute node;
- place UV, Hugging Face, Torch, Triton, CUDA, W&B, Matplotlib, pip, XDG, and temporary caches under SLURM_TMPDIR;
- delete the job-local environment and caches at exit;
- persist only trajectories, checkpoints, edited images, metrics, and logs;
- avoid persistent cache or virtual-environment directories in the repository or shared scratch;
- use ordinary jobs rather than a throttled array when array limits would serialize the work;
- use afterok dependencies for metrics or continuation stages;
- never submit the full matrix before smoke tests pass.

Do not use legacy PLGrid paths, account names, partitions, modules, or cache locations.

## Validation gates

Do not advance past a gate merely because jobs disappeared from squeue.

### Gate A: configuration

- Hydra composes for SD1.4 gather, standard training, and all editor evaluations.
- Resolved configs contain CompVis/stable-diffusion-v1-4 and SD1.4 output roots.
- No resolved SD1.4 config contains runwayml/stable-diffusion-v1-5 or an SD1.5 trajectory/checkpoint path.

### Gate B: trajectory smoke

Generate two prompts and verify all files, shapes, dtypes, timestep ordering, target ordering, and deterministic rerun behavior.

### Gate C: full trajectory audit

Verify the complete 30,441-sample root and save the audit report.

### Gate D: training smoke

For every distinct loss/sampler implementation, run at least one optimizer update with a real SD1.4 UNet and real cached trajectories. Check finite component losses, finite gradients, adapter updates, and checkpoint reload.

### Gate E: editor smoke

For one PIE-Bench sample, test all six LoRA/editor paths. Confirm model ID, checkpoint loading, output files, and that pure LoRA outputs do not call Direct Inversion.

### Gate F: full completion

For every Slurm job, inspect sacct state, exit code, elapsed time, stderr, expected image count, and expected metric files. COMPLETED with a missing artifact is a failed pipeline.

## Completion checklist

- [ ] Record Git commit and target-HPC environment.
- [ ] Verify ReCap manifest count and SHA-256.
- [ ] Add and compose SD1.4 Hydra configs.
- [ ] Pass two-sample trajectory smoke.
- [ ] Generate and audit all 30,441 CFG1 FP32 trajectories.
- [ ] Reproduce six SD1.4 no-LoRA controls.
- [ ] Train standard uniform r8 and r16.
- [ ] Train standard final-tail r8 and r32.
- [ ] Evaluate pure LoRA and fused LoRA plus DirectInv with all three editors.
- [ ] Calculate metrics immediately after successful generation.
- [ ] Save per-image metrics and aggregate summaries.
- [ ] Produce the SD1.4 comparison table.
- [ ] Report failures, missing artifacts, exact checkpoints, and exact output paths.
- [ ] Only then consider optional CFG2.5 r8 or additional ranks.
