# Finalized opposite-reader source integration

Imported selectively from `vision/wikiart-opposite-prompt` commit
`6f152880335bab992487f5119cddb5572c5f3b07`, which includes synthetic recovery
`985a640` and `2a30110`, WikiArt preparation `decb039`, and the strict CSD
direct/wrapped vision-tower fix. No old branch merge, training changes,
lockfile changes, generated results or model payloads were imported.

`diff_inversion/steering/lora.py` and `sampling.py` preserve the source branch's
helpers locally. Steering imports and tests use these helpers so current
editor/training behavior is not replaced. The final diagnostic implementation
accepts a fresh sibling comparison directory while rejecting overlap with run
inputs, and rejects non-finite vectors.

## External evidence and runtime prerequisites

Set `VISION_SD15_EVIDENCE_DIR` to the existing scratch/persistent directory
containing `preflight.json`, `unet_config.json`, and `input_evidence.json` from
the validated preparation. Evidence is deliberately not a Git payload. The
runtime still checks model identity and every expected attention target/shape.
For this integration's PW metadata-only checks, evidence copies reside at:
`/scratch/aszymczy/projects_cache/Diffusion-Inversion-for-Image-Editing/data/processed/source-sync-20261112/sd15-evidence`.

The imported configs and launchers preserve the original protocol and historical
absolute artifact references. Before a new run, prepare its immutable config,
commit/hashes, fresh scratch output and external evidence binding; old preparation
commit guards must not be bypassed. These source files alone do not authorize a run.
Use `run_outline_reader_recovery_2.sh` for the finalized synthetic recovery;
recovery_1 is retained as historical source. WikiArt recovery reuses completed
stages and includes scoring and summaries.

The legacy transfer/partial-inversion launchers committed separately retain
historical checkout-relative reports/log destinations and /tmp fallbacks. They
require storage-policy review before execution. No launcher was executed here.

## Validation scope

System-Python AST and shell syntax checks; ten Direct Inversion test bodies;
three OVG seed-summary unit tests; SD1.5 metadata and negative checks;
recovery output guards; CSD direct/wrapped strict-load dispatch using stand-ins.
No GPU/model loading or dependency installation. The project .venv target was
unavailable, so Torch/pytest/Pillow-dependent tests remain unrun. Checkpoints,
datasets, science results and manuscript sources were left untouched.
