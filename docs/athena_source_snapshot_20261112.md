# Athena source snapshot for PW integration

This topic preserves existing Athena work on base 8694649c07685a7ab3c752e558392460f4a0cd58:
source-mixture prompt preparation/dataset metadata, frozen PIE labels and OOD prompts,
SD1.5/SDXL cycle metrics, model/cache overrides, graceful checkpoint handling,
launchers, tests, dependency lock and six publication TeX sources (one modified).
It additionally imports portable DCI storage/validation/extraction utilities.
No generated metrics/images, local absolute symlinks, credentials, Vision metadata,
or divergent historical steering branch are included. PW owns integration into
img_edition; reconcile overlaps there instead of merging historical steering ancestry.

Existing science and launcher behavior are preserved, not certified by a snapshot:
- Mixture configuration uses microbatch64/accumulation2, despite the earlier
  AGENTS experiment outline specifying batch128/accumulation1.
- Prompt selection derives per-stratum/review seeds from42 rather than using the
  identical seed42 for every selection in the original outline.
- Historical launchers contain raw scheduler calls, a polling watcher, site defaults,
  long resource requests, and model staging without immutable revisions. They are
  archival source in this commit, not authorized Vision entrypoints. Future execution
  requires an assigned task and the shared scheduler adapter; none ran here.
- Some configs still default output paths beneath reports/. Future runs must override
  these to helper-selected storage and explicitly configure scratch caches/temp.
- Publication values are preserved source claims, not revalidated results. The SDXL
  r32 held-out table does not bind PSNR30.40/LPIPS.06 to an exact checkpoint step.

The main working tree/index/branch remain unchanged; the topic is built with an
isolated Git index. Local artifact manifests contain the complete source path/hash
list and validation evidence. DCI utility changes only remove site-specific paths,
recheck the archive digest, and make extraction status observation-based.
