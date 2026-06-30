# Diffusion Inversion for Image Editing

This project improves DDIM inversion for Stable Diffusion XL and SD1.5 by
fine-tuning a lightweight LoRA adapter on the diffusion UNet. The adapter is
trained to produce a better inversion-time approximation of the frozen base
model's noise prediction at the next, noisier latent state:

```text
eps_phi(x_{t-1}, t, c) ~= eps_theta(x_t, t, c)
```

The base model stays frozen. Only LoRA parameters injected into UNet attention
projections (`to_q`, `to_k`, `to_v`, `to_out.0`) are trained. The adapter can
be enabled only for inversion, leaving normal sampling unchanged unless the
LoRA is explicitly activated.

## Project Layout

```text
diff_inversion/
  data/
    generate_sdxl_samples.py       # Generate SDXL/SD1.5 images and trajectories
    latent_trajectory_dataset.py   # Dataset of latent transitions for training
    precompute_training_cache.py   # Add conditioning.pt and target_eps.pt
  modeling/
    train.py                       # Train UNet LoRA
    sdxl_sampling.py               # Shared DDIM sampling/inversion helpers
    validation_preview.py          # Optional previews during training
  eval/
    invert_sdxl.py                 # DDIM inversion, optionally with LoRA
    pnp_lora.py                    # Submit PnP editing with SD1.5 LoRA inversion
    reconstruct_sdxl.py            # Reconstruct images from inverted_noise.pt
    run.py                         # Metrics and preview report
    reporting.py                   # JSON/CSV/Markdown/W&B output helpers

config/
  model/sdxl.yaml                  # SDXL base, DDIM, 50 steps, 1024x1024
  model/sd15.yaml                  # SD1.5 base, DDIM, 50 steps, 512x512
  sample_gather*.yaml              # Train/val/test trajectory generation
  precompute_training_cache*.yaml  # Cached training target generation
  train/sdxl_lora*.yaml            # LoRA variants: r8/r16/r32
  train/sd15_lora*.yaml            # SD1.5 LoRA training configs
  eval/*.yaml                      # Inversion, reconstruction, and reporting

scripts/
  launch_sample_gather_submitit.sh
  launch_sd15_sample_gather_submitit.sh
  launch_sample_gather_eval_submitit.sh
  launch_precompute_training_cache_{train,val,test}_submitit.sh
  launch_sdxl_lora_train_submitit.sh
  launch_sd15_lora_train_submitit.sh
  launch_sdxl_eval_{invert,reconstruct,run}_submitit.sh
  launch_pnp_lora_eval_submitit.sh
  submit_sdxl_eval_chain.sh
```

## Environment

The project targets Python 3.11 and `uv`. Dependencies are defined in
`pyproject.toml`; PyTorch is configured for CUDA 12.8 through the PyTorch CUDA
index.

```bash
uv sync --frozen
```

Some scripts and configs contain machine-specific default paths. Before running
them, check `REPO_DIR`, `PYTHON`, `OUTPUT_DIR`, `input_dir`, `root_dir`, and
`checkpoint_dir`, or override them with environment variables / Hydra overrides.

## Data Format

Each generated example is stored in a `sample_XXXXXX` directory:

```text
sample_000000/
  final.png
  prompt.json
  meta.json
  timesteps.json
  latents/
    trajectory.pt
  conditioning.pt
  targets/
    target_eps.pt
```

`trajectory.pt` stores the full latent trajectory as one stacked tensor. For
SDXL 1024x1024 images, one latent has shape `4 x 128 x 128`; for SD1.5
512x512 images, one latent has shape `4 x 64 x 64`.

`conditioning.pt` stores UNet conditioning. SDXL samples include:

```text
prompt_embeds
pooled_prompt_embeds
add_time_ids
```

SD1.5 samples store only:

```text
prompt_embeds
```

`target_eps.pt` stores cached teacher noise predictions from the frozen base
UNet. This avoids recomputing the teacher target during every LoRA training
step.

## Data Preparation

Prompts come from Recap-COCO (`config/data/recap_coco.yaml`). Trajectory
generation uses `StableDiffusionXLPipeline` with
`stabilityai/stable-diffusion-xl-base-1.0`, DDIM scheduling, 50 inference
steps, guidance scale `1.0`, and 1024x1024 resolution.

Generate train trajectories:

```bash
scripts/launch_sample_gather_submitit.sh
```

Generate val/test trajectories:

```bash
scripts/launch_sample_gather_eval_submitit.sh
```

The default job specs split train generation into 8 chunks and eval generation
into separate val/test chunks.

SD1.5 uses the same generation and training code with
`StableDiffusionPipeline`, `runwayml/stable-diffusion-v1-5`, 512x512 images,
and prompt embeddings only:

```bash
scripts/launch_sd15_sample_gather_submitit.sh
scripts/launch_sd15_val_sample_gather_submitit.sh
scripts/launch_sd15_test_sample_gather_submitit.sh
```

If trajectories already exist but `conditioning.pt` and `targets/target_eps.pt`
are missing, add the training cache without regenerating images:

```bash
scripts/launch_precompute_training_cache_train_submitit.sh
scripts/launch_precompute_training_cache_val_submitit.sh
scripts/launch_precompute_training_cache_test_submitit.sh
```

## LoRA Training

Training is implemented in `diff_inversion/modeling/train.py`. The dataset
treats each saved trajectory as a set of latent transitions. For transition
`i`, the loader returns:

```text
x_clean             # cleaner latent used as the student input
timestep            # target timestep
prompt_embeds
pooled_prompt_embeds # SDXL only
add_time_ids         # SDXL only
target_eps           # eps_theta for the next/noisier latent state
```

Training loss:

```text
MSE(student_eps, target_eps)
```

LoRA configs:

```text
config/train/sdxl_lora.yaml      # r=16, alpha=8
config/train/sdxl_lora_r8.yaml   # r=8,  alpha=4
config/train/sdxl_lora_r32.yaml  # r=32, alpha=16
config/train/sd15_lora.yaml      # SD1.5 r=16, alpha=8
```

Shared training settings:

```text
learning_rate: 5e-5
lr_scheduler: cosine
warmup_steps: 1000
batch_size: 2
gradient_accumulation_steps: 4
gradient_checkpointing: true
max_grad_norm: 1.0
save_every_steps: 5000
save_training_state: true
```

Run the default r16 variant:

```bash
scripts/launch_sdxl_lora_train_submitit.sh train=sdxl_lora
```

Run comparison variants:

```bash
scripts/launch_sdxl_lora_train_submitit.sh train=sdxl_lora_r8
scripts/launch_sdxl_lora_train_submitit.sh train=sdxl_lora_r32
```

Run SD1.5 LoRA training:

```bash
scripts/launch_sd15_lora_train_submitit.sh
```

For SD1.5 rank variants, pass rank overrides directly:

```bash
scripts/launch_sd15_lora_train_submitit.sh lora.r=8 lora.lora_alpha=4
scripts/launch_sd15_lora_train_submitit.sh lora.r=32 lora.lora_alpha=16
```

Checkpoints are saved as:

```text
checkpoint_step_*.pt        # LoRA adapter weights only
training_state_step_*.pt    # LoRA + optimizer + scheduler + RNG state
checkpoint_final.pt
training_state_final.pt
```

Resume from full training state:

```bash
scripts/launch_sdxl_lora_train_submitit.sh \
  train=sdxl_lora \
  resume.enabled=true \
  resume.mode=training_state \
  resume.checkpoint_path=/path/to/training_state_step_30000.pt
```

Resume from adapter weights only. This restores the LoRA weights and advances
the scheduler to the requested step, but optimizer moments are not restored:

```bash
scripts/launch_sdxl_lora_train_submitit.sh \
  train=sdxl_lora \
  resume.enabled=true \
  resume.mode=adapter \
  resume.checkpoint_path=/path/to/checkpoint_step_15000.pt \
  resume.global_step=15000
```

## Evaluation

Evaluation has three stages:

1. `invert_sdxl.py` runs DDIM inversion from the final image latent to
   `inverted_noise.pt`; LoRA can be enabled here.
2. `reconstruct_sdxl.py` reconstructs `reconstructed.png` from
   `inverted_noise.pt`; LoRA is disabled by default to isolate the adapter's
   effect on inversion.
3. `run.py` computes metrics and writes the report.

Run the full sequential eval chain for one LoRA variant:

```bash
scripts/submit_sdxl_eval_chain.sh r8
scripts/submit_sdxl_eval_chain.sh r16
scripts/submit_sdxl_eval_chain.sh r32
```

The chain script creates a separate output directory and runs
inversion -> reconstruction -> report in order.

PnP editing can run with SD1.5 LoRA-assisted inversion through the
`lora+pnp` method. The wrapper loads the LoRA checkpoint into
`pnp_inversion/run_editing_pnp.py`, enables it only for inversion, and then
runs Plug-and-Play editing:

```bash
scripts/launch_pnp_lora_eval_submitit.sh \
  lora.r=16 \
  lora.lora_alpha=8 \
  lora.checkpoint_path=/path/to/checkpoint_step_15000.pt
```

Override input/output locations:

```bash
INPUT_DIR=/path/to/test \
EVAL_ROOT=/path/to/reports/eval \
RUN_ID=my_run \
scripts/submit_sdxl_eval_chain.sh r16
```

If inversion and reconstruction artifacts already exist, rerun only the report:

```bash
scripts/launch_sdxl_eval_run_submitit.sh \
  input_dir=/path/to/sdxl_trajectories_stacked/test \
  artifact_dir=/path/to/eval_run/artifacts \
  output_dir=/path/to/eval_run \
  inverted_noise_name=inverted_noise.pt \
  reconstructed_image_name=reconstructed.png \
  save_noise_previews=false \
  save_image_previews=true \
  image_preview_max_samples=32 \
  save_normality_plots=false \
  calculate_lpips=false \
  inversion_diagnostics.enabled=false \
  inversion_diagnostics.save_plots=false
```

Report outputs:

```text
evaluation_summary.json
evaluation_summary.md
noise_comparisons/noise_comparisons.{json,csv}
normality/normality_comparisons.{json,csv}
image_comparisons/image_comparisons.{json,csv}
image_comparisons/*.png
inversion_diagnostics/*       # if diagnostics are enabled
```

Main metric groups:

```text
aggregate/initial_vs_inverted_noise/*
aggregate/inversion_error_stats/*
aggregate/final_vs_reconstructed_image/mse|rmse|mae|psnr_db|ssim
aggregate/initial_vs_inverted_noise_normality/*
```

## Documentation

Project documentation:

```text
reports/ZZSN_Dokumentacja_Wstepna.pdf
reports/ZZSN_Dokumentacja_Końcowa.pdf
```
