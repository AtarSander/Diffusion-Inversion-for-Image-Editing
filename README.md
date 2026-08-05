# Diffusion Inversion for Image Editing

This project trains LoRA adapters to improve diffusion inversion and evaluates
them with P2P, PnP, MasaCtrl, and Pix2Pix-Zero. Each editor supports:

- `lora+<editor>`: LoRA inversion followed by the standard editor.
- `lora+directinversion+<editor>`: LoRA inversion followed by Direct
  Inversion's trajectory correction and then editing.

## Environment

The project uses Python 3.11 and `uv`:

```bash
uv sync --frozen
```

Cluster jobs use `gpu-l` on `h86`, create a job-local virtual environment,
and write Slurm stdout/stderr to `logs/slurm/`.

W&B credentials belong in the ignored repository `.env`:

```dotenv
WANDB_API_KEY=...
```

Training loads this file automatically. The secret is passed through the
environment and is never stored in Hydra's resolved configuration.

## Configuration

Hydra settings are organized by concern rather than by experiment matrix:

```text
config/
  model/
    sd15.yaml
    sdxl.yaml
  data/
    sd15_trajectories.yaml
    sdxl_trajectories.yaml
  lora/
    base.yaml
    r8.yaml
    r16.yaml
    r32.yaml
  wandb/
    online.yaml
    offline.yaml
    disabled.yaml
  lr_scheduler/
    cosine.yaml
    constant.yaml
  train/
    base.yaml
    sd15.yaml
    sdxl.yaml
  eval/
    lora_edit.yaml
    p2p_lora.yaml
    pnp_lora.yaml
    masactrl_lora.yaml
    pix2pix_zero_lora.yaml
    pnp_lora_metrics.yaml
  train_sd15.yaml
  train_sdxl.yaml
  sample_gather_sd15.yaml
  sample_gather.yaml
  precompute_training_cache.yaml
```

`train_sd15.yaml` only composes independent choices:

```yaml
defaults:
  - model: sd15
  - data: sd15_trajectories
  - lora: r16
  - wandb: online
  - lr_scheduler: cosine
  - train: sd15
```

Precision is an ordinary model setting:

```yaml
model:
  torch_dtype: float32
  variant: null
```

LoRA architecture variants live in their own files. Checkpoints, output paths,
and inversion guidance remain ordinary runtime overrides.

## Trajectory data

The SD1.5 gather config reads all 30,441 Recap-COCO prompts and writes one
unified dataset:

```text
/scratch/aszymczy/projects_cache/Diffusion-Inversion-for-Image-Editing/
  data/processed/sd15_trajectories_stacked_fp32/all/
```

Each sample contains:

```text
sample_000000/
  final.png
  prompt.json
  meta.json
  timesteps.json
  latents/trajectory.pt
  conditioning.pt
  targets/target_eps.pt
```

Generate trajectories:

```bash
sbatch scripts/slurm/generate_sd15_trajectories_fp32_h86.sbatch
```

The generator uses batch size one per prompt. The Slurm array partitions the
prompt indices; all tasks write to the same dataset directory.

## LoRA training

The current SD1.5 run settings are in `config/train/sd15.yaml`; LoRA,
scheduler, W&B, model, and dataset settings come from their respective groups:

```yaml
max_train_steps: 35676       # three epochs
gradient_accumulation_steps: 1
learning_rate: 5e-5
model:
  torch_dtype: float32
  variant: null
```

Submit rank 8, 16, or 32:

```bash
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 8
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 16
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 32
```

The launcher selects `lora=r8`, `lora=r16`, or `lora=r32`; the corresponding
file defines both rank and alpha. It also gives each run a separate checkpoint
directory. Additional Hydra choices can follow the rank:

```bash
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 8 wandb=offline
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 32 lr_scheduler=constant
```

Training checkpoints include:

```text
checkpoint_step_*.pt
training_state_step_*.pt
checkpoint_final.pt
training_state_final.pt
```

## Final-step oversampling

Training defaults to `sampling=uniform`. To oversample the final DDIM
trajectory transitions while keeping each epoch the same length, select:

```bash
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 8 sampling=final_tail
```

The `final_tail` profile makes the final 10% of transitions from every
trajectory account for 50% of training draws. Both fractions are configurable:

```bash
sbatch scripts/slurm/train_sd15_lora_h86.sbatch 8 \
  sampling=final_tail \
  sampling.final_step_fraction=0.2 \
  sampling.target_draw_fraction=0.5
```

Validation remains uniformly sampled in both modes.

## Editing evaluation

`config/eval/lora_edit.yaml` contains shared paths. Evaluation selects the same
`lora=r8/r16/r32` group used by training; the four method configs only select
their editor-specific method names.

Run all pure and Direct-Inversion-fused variants for an existing rank:

```bash
sbatch scripts/slurm/eval_lora_h86.sbatch 8
sbatch scripts/slurm/eval_lora_h86.sbatch 16
sbatch scripts/slurm/eval_lora_h86.sbatch 32
```

The eight array indices are:

```text
0  lora+p2p
1  lora+directinversion+p2p
2  lora+pnp
3  lora+directinversion+pnp
4  lora+masactrl
5  lora+directinversion+masactrl
6  lora+pix2pix-zero
7  lora+directinversion+pix2pix-zero
```

To evaluate only fused methods with inversion guidance 2.5:

```bash
sbatch --array=1,3,5,7 scripts/slurm/eval_lora_h86.sbatch 32 2.5
```

## Metrics

Calculate all eight editing metrics for one rank:

```bash
sbatch scripts/slurm/eval_lora_metrics_h86_all_ranks.sbatch 8
sbatch scripts/slurm/eval_lora_metrics_h86_all_ranks.sbatch 16
sbatch scripts/slurm/eval_lora_metrics_h86_all_ranks.sbatch 32
```

Metrics and summaries are written below
`/scratch/aszymczy/projects_cache/Diffusion-Inversion-for-Image-Editing/evaluation/metrics/`.

## Main modules

```text
diff_inversion/data/generate_sdxl_samples.py
diff_inversion/data/latent_trajectory_dataset.py
diff_inversion/data/precompute_training_cache.py
diff_inversion/modeling/train.py
diff_inversion/eval/{p2p,pnp,masactrl,pix2pix_zero}_lora.py
diff_inversion/eval/pnp_lora_metrics.py
```

Project reports are under `reports/`.
