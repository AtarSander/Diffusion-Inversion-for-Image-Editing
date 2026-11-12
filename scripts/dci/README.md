# DCI acquisition utilities

These portable copies preserve the Athena acquisition helpers. They do not download
images, accept SA-1B terms, install packages, or run models. Inspect the site's
`mkdata`/`mkvenv` definitions and existing links before selecting storage; check
filesystem bytes/inodes AND per-user quota. Never select home or a source checkout.

Set `DCI_PROJECT_DATA` to the resolved helper data root, `DCI_QUOTA_MOUNT` to its
Lustre mount, and optionally `DCI_ACQUISITION` beneath that data root, then source
`scripts/dci/storage_env.sh`. No site path or credential is embedded. The extractor
requires `lfs quota`; unsupported quota backends fail closed and need a site adapter.

Both Python commands require `$DCI_ACQUISITION/annotation_download.json` containing
`verified: true`, `archive_path`, `expected_sha256`, and `actual_sha256`. Obtain the
expected digest from the pinned official DCI repository, not from an untrusted
archive. The scripts independently hash archive bytes before processing.

- `python scripts/dci/validate_dci_archive.py`: stream official annotation JSON,
  preserve captions/split IDs, inventory photos and verify/decode available images.
  Outputs are under `$DCI_PROJECT_DATA/raw/DCI/metadata`; Pillow is needed only when
  photos exist. Use a fresh metadata destination: outputs are not a resume protocol.
- `python scripts/dci/safe_extract_dci.py`: reject traversal, duplicate names,
  links and special files; require filesystem and user inode headroom; extract
  through `extracted.part` and rename to `dataset`. Existing destinations are refused.
  Interrupted partial directories remain for explicit inspection, never automatic deletion.

Official source: https://github.com/facebookresearch/DCI at
8326c972847f88b45a01d3f553930558bf60a136. Its annotation archive SHA256 is
`d865c244150168d3f25daaad0bf5b70b2123cf3e83ca7b0e207d5b26943c5fc7`.
Annotation archives contain no raw photos. Obtain only the required SA-1B shard
through official licensed access; preserve official names and splits. This import
makes no assertion that images are available or reconstruction is ready.
