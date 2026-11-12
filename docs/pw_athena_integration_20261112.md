# PW/Athena source integration, 2026-11-12

Athena snapshot `840c3ac98149b0e76a3494f40fd73177be149ca0` was imported
from a complete-history bundle verified with Git and SHA256
`5c30586f290e92a5e23636447eb54d89cb5455268d481d530a9d401eb0872262`.
It was cherry-picked onto PW `d8d56c7` as `8587f23`.

## Reviewed overlaps

- Keep both `save_intermediates` and `save_inversion_artifacts`; either false
  suppresses intermediate buffers/files. Retain Athena model-ID override.
- Keep PW PnP empty reference prompt and DreamSim lazy import/scratch cache handling.
- Keep PW endpoint sampling and finite/non-negative source-weight guards, source
  metadata and full-path default source names. Add Athena unique-name validation.
  Do not leave duplicate source-mixture methods or metadata keys after merging.
- Keep both sampler test suites; Athena's synthetic fixture explicitly selects
  transition mode. Keep Athena graceful-stop checkpoint logic and source metadata.
- PW steering-local helpers, finalized recovery and strict CSD loading are unchanged.

## Checks and limits

Five offline DCI fixtures passed. All changed Python AST/shell syntax and 21 YAML
configs checked (see execution report for actual count). Source-mass/invalid-weight/
endpoint checks used AST-extracted logic with NumPy storage, not a Torch test run.
All four artifact-flag combinations passed. Existing SD1.5 metadata/negative checks
and three OVG summary tests passed. No GPU inference, datasets, downloads or installs.
Full ML-dependent suites remain unrun because the existing project env is unavailable.

Publication TeX values were preserved, not reanalysed. Athena snapshot guidance
records historical protocol deviations and archival launcher limitations; integration
is not authorization to run those scripts. Generated local metrics/images and Vision
metadata remain excluded. The two requested PW manuscript sources are committed
separately as `2a0b6d2`.
