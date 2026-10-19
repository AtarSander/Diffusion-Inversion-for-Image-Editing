# Image-derived SDXL steering

One configurable pipeline for collecting activations from real images, fitting
contrastive directions, and comparing steered generations against their own base.
DDIM and Moonwalker (MW) name the **source inversion methods**. Both sets of
directions steer the same base SDXL model with LoRA disabled.

| Module | Responsibility |
|---|---|
| `images.py` | Load SDXL/LoRA, encode text, deterministic VAE mode, decode images |
| `runtime.py` | Invert images, read saved states or free reconstructions, generate |
| `hooks.py` | Capture CA/SA/block updates and inject conditional channel offsets |
| `directions.py` | Differences of means, null directions, layer selection and dose |
| `config.py` | Read YAML and the explicit construction-source list |
| `experiment.py` | Connect the three stages and save results with prompts/settings |
| `__main__.py` | The common command-line interface |

Experiments vary through configuration and source lists. There are no per-run
Python scripts, source-code inventories, configuration fingerprints, cluster
launchers or environment audits in this package.

## Method

For every construction image, invert the same VAE latent using DDIM and MW.
MW enables the existing inversion LoRA only during inversion; the adapter is
disabled during DDIM inversion, feature reading and target generation.

The reader records the conditional activations at every denoising call. Set
`reader: free` for a base-model reconstruction starting at recovered `zT`, or
`reader: direct` for independent probes of the corresponding saved inverse states.
Those reads answer different questions and belong in separate comparisons.

All three representations are recorded during the same forwards:

- **SA**: `attn1` output after its projection, before its residual addition.
- **CA**: `attn2` output after its projection, before its residual addition.
- **block**: `BasicTransformerBlock` output minus input, subtracting the actual
  tensors in FP32. The block includes SA, CA, feed-forward layers, normalization
  and their residual connections. Injection happens at its output. It is not
  an entire UNet up block or the outer `Transformer2DModel`.

Token means form one channel vector per image, layer and denoising step. For
each method independently:

```text
d[representation, method, layer, step] = mean(positive) - mean(negative)
offset = lambda * sqrt(10 / (L*K)) * reference_RMS * d / max(RMS(d), 1e-8)
```

`reference_RMS` uses unpooled full-map second moments from the construction
images, pooled equally over DDIM/MW. It is separate for every representation,
layer and step. `L` and `K` count selected layers and active calls. This dose
normalization does not establish equal perceptual effects across representations.

The direction bundle contains `true`, balanced paired sign-swap (`swap`) and
shared Gaussian (`gaussian`) orientations. This interface uses equal, even class
sizes of at least two to support the balanced sign control. Positive/negative
rows are paired in their respective file orders. Merely permuting negative
partners would leave the difference of means unchanged and is not this null.

Vectors are **time-dependent**. A vector read at one timestep is not silently
reused at other timesteps. The first stored row is denoising call 0, rather than
the clean endpoint or inversion step 0.

## Run

Use the repository's Python 3.11 environment and dependencies from `pyproject.toml`:

```bash
uv sync --frozen --group dev
```

Start with [the example YAML](../../config/steering/van_gogh.yaml). Supply your
local SDXL model and native tensor-only inversion LoRA checkpoint. The example
uses rank 16 / alpha 8, 50 DDIM steps, 1024 pixels, neutral `A painting` source
conditioning and CFG 1. It describes the cleaned interface, not a newly executed
experiment or a measured optimum.

Set `source_manifest` to a JSONL containing the chosen **construction images**:

```jsonl
{"id":"vg_001","path":"images/vg_001.png","label":"positive"}
{"id":"vg_002","path":"images/vg_002.png","label":"positive"}
{"id":"other_001","path":"images/other_001.png","label":"negative"}
{"id":"other_002","path":"images/other_002.png","label":"negative"}
```

These four rows illustrate the format, not the historical pilot selection.
Use the actual chosen sources; keep validation/test images out of this list.
Paths in YAML are relative to that YAML; image paths are relative to the JSONL.
The model identifier may be a Hugging Face model ID or a local model directory.
Copyrighted source images, model weights and generated artifacts are not bundled.

```bash
python -m diff_inversion.steering extract \
  --config config/steering/van_gogh.yaml --output reports/steering/source_v1

python -m diff_inversion.steering fit \
  --features reports/steering/source_v1/features.pt --output reports/steering/vectors_v1

python -m diff_inversion.steering generate \
  --config config/steering/van_gogh.yaml \
  --vectors reports/steering/vectors_v1/directions.pt --output reports/steering/generation_v1
```

Each output directory must be new. Feature/direction files use the new package's
format; historical HPC artifact bundles need explicit conversion. Changing the
source reader, prompts, model, LoRA or diffusion schedule requires new extraction.

## Controlled comparisons

Use `generation.representations`, `strengths` and `controls` for a common matrix
of variants. Select `layers: all`, `last36` or `last12`; the latter two use the
last up-path blocks in observed CA execution order. On SDXL Base these correspond
to blocks 34–69 and 58–69 in the historical 70-block ordering. A block
representation does not imply injection into every layer.

`start_call` is inclusive and `stop_call` exclusive, both zero-based. For DDIM50,
`[1, 50)` means ordinal denoising steps 2–50. To compare layer/time profiles,
change those YAML values and choose a new output directory; reuse the same
vectors, prompts and seeds.

Keep one primary seed across **all prompts and variants**. Additional seeds
repeat the complete comparison. The exact same starting-noise tensor goes to
the base, DDIM-vector and MW-vector generations within each unit; tensors are
also saved. The output manifest records prompts, seeds, variants, layer/time
selection and the starting-noise hash. This is the only hash in this path and
serves the requested paired-generation comparison.

Outputs are grouped as `seed_<seed>/prompt_<index>/`, with `base.png` and readable
variant filenames. Source reconstruction PNGs are saved by `extract` in free
reader mode. `features.pt` stores pooled activations and moments; `directions.pt`
stores raw directions, controls and reference RMS. Full hidden maps are not saved.

## Validation and results

```bash
python -m pytest -q tests/test_steering_directions.py \
  tests/test_steering_runtime.py tests/test_steering_experiment.py
```

Tests use actual small Diffusers transformer blocks and schedulers without
pretrained weights. They cover the hook boundaries, conditional-only addition,
time indexing, LoRA routing, common dose, nulls and the complete three-stage flow.
Full SDXL image parity after this refactoring has not been run. Semantic scoring,
automatic source curation, training and HPC submission remain outside this path.

See [the code review](REVIEW.md) and the
[historical Van Gogh summary](../../docs/weekly/2026-10-12_2026-10-18_van_gogh_steering.md).
That summary refers to local result artifacts, which are not part of the Git
commit. None of the refactoring tests is new evidence of style transfer.
