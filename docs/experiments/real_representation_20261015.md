# CA, SA and block updates at inverted real-image latents

2026-10-15. Source-only follow-up authorized by the user's request to continue
work on existing-image inversion. Athena **3368475 COMPLETED 0:0**,
15:22:01–15:40:19 CEST, 2401calls/48reconstructions. All85remote artifact
hashes and all48PNG/final-latent parent parities passed. Independent CPU replay
reconstructed all21000diagnostic rows/252000scalar comparisons exactly.
Decoder36/calls1–49 AUC DDIM/MW: CA .65157/.67428, SA .86193/.85420,
block update .85594/.84584. This is better cohort separation, not causal style
steering or activation-ID evidence. See the
[completed readout](../../reports/technical/real_representation_20261015_v1/FINAL_READOUT.md).

## Actual execution contract

PlanSHA `01193d482e6623c6b308db893098bbe4ac8da9a2756afa0bd8c9e53d61f98d6b`;
snapshot `wikiart-steering/code_20261015_real_representation_v1`,93files,
inventorySHA `81755d4797fdde5497dc38c5f5730789c998df35c98c646c55df4eb0ab1967b9`.
All81 inherited follow-up payloads and4 historical free-reader additions match
their sealed predecessors. Eight new payloads comprise4production modules,
3test modules and this protocol at freeze time. No historical worker changed.

All82 focused tests passed locally118.18s and onAthena146.61s, with0skips;
Ruff passed. Independent plan review verified464input hashes, all93snapshot
payloads, the exact24sources/48cases and resource/conditioning contract. Full
19682-file numerical runtime checks ran in preparation/submission and remain
required before/after worker execution. The actual83200-channel storage bound
is3,067,804,176B under4GiB. Initial automatic review misread the old proposal
status as a prohibition; renewed review of the same action accepted the latest
user instruction and bounded implementation. No approval issue remains open.

[Submission](../../reports/technical/real_representation_20261015_v1/submission.json),
[exact plan](../../reports/technical/real_representation_20261015_v1/plan.json),
[independent plan audit](../../reports/technical/real_representation_20261015_v1/independent_plan_audit.json).

The separate read-only observer and offline analyzer passed28tests/Ruff and peer
review. They preserve all21000diagnostic cells, verify all new output hashes and
keep26large tensor artifacts remote. Identical reconstruction PNGs may hardlink
to the previously verified parent pixels. Prior visual review is inherited only
after independent all48PNG hash parity. These tools do not independently rehash
all inherited model/runtime bytes; worker full pre/post guards remain necessary.
The stronger independent raw-source CPU replay subsequently ran successfully:
all24new/24parent caches, raw/null directions, RMS, heldout scores and all
diagnostic rows were verified with maximum error zero. It read7.53GB remotely,
without model/GPU calls or tensor downloads. ReceiptSHA
`40023f1b75f1619d694676f9003308056bb99ae0c0373b50e59319e6c22498c7`.
No automatic causal target job was attached. A
[separate existing-image steering pilot](real_representation_steering_20261015.md)
is now being prepared after readback and the user's renewed continuation request.

## Question

Does the representation read from an inverted real image affect the stability
and heldout usefulness of its image-defined class contrast? Compare CA output,
SA output and the update of the same BasicTransformerBlock. This follows the
[updated research priority](real_still_life_casteer_20261015.md), without a new
native generation or prompt-vector validation prerequisite.

The completed free-reader diagnostic3366199 is the numerical reference. It
reconstructed24 reviewed artworks from their existing DDIM/Moonwalker endpoints;
CA free-reader heldout separation did not improve over the saved-state reader.
Better reconstruction, class separation and causal style steering are distinct.

## Fixed data and trajectory

- Same24 real artworks:16 construction and8 validation, unchanged roles and IDs.
- Same sealed DDIM/MW recovered xT; no fresh inversion or source selection.
- Base SDXL, DDIM50,1024px,FP16, literal `A painting`, empty negative, CFG1.
- The historical Moonwalker r16/alpha8/step2500 adapter remains attached through
  the inherited loader but is disabled for every feature read and reconstruction.
- Each reconstruction evolves freely from xT. Never substitute saved inverse
  states into later denoising calls. Both methods use the same base model.
- No random-noise target generation, steering intervention, training or learned
  style-encoder inference in this stage.

## Simultaneous instrumentation

Match the70 actual BasicTransformerBlocks already indexed by their attn2 modules.
Collect all50 calls at all70 locations for every source and recovery method:

| Representation | Definition |
|---|---|
| CA | attn2 output after output projection, before external residual addition |
| SA | attn1 output after output projection; require only_cross_attention=False |
| Block update | FP32(block output) minus FP32(block input), before spatial pooling |

Keep one conditional branch and the same spatial mean for all readers. Record
full-map second moments using the specified producer precision. The block input
must be retained without mutation until its output is read; release full maps
immediately after pooling. Do not retain full spatial maps across calls.

The proposed block boundary excludes the outer Transformer2D projections and
outer residual. It is not identical to the Transformer2D-level boundary named by
the SD1.4 SAE paper. Likewise, an SA output vector is not SA K/V replacement.

Read-only hooks must preserve outputs, RNG, scheduler and adapter state, and clean
up on exceptions. Reproduce the parent CA feature arrays, second moments, final
latents and PNG bytes for all48 cases before accepting the new publication.
Any parity failure is a technical failure, not a negative representation result.

## Construction and evaluation

Fit separate class-mean differences for every representation, method, layer and
call using construction only. Compute reference RMS from construction full-map
second moments, giving the two recovery methods equal weight within each
representation. Scales are diagnostic here: no activation is modified.

Preserve the fixed first4 versus last4 construction-pair split for consistency
with the parent. This split is subject/composition dependent, not a random-half
reliability estimate. Project the same8 validation artworks on each construction
direction and report the complete score/AUC tables. No validation fitting,
representation/layer/window selection or independent-test claim.

Retain all cells and prespecified descriptive bins: all70, last36 and the original
single-up module; call0, early1–9, middle10–39 and late40–49. Layer/time cells share
sources and are correlated. Summaries cannot turn3500 cells into3500 independent
artworks. Compare feature stability within each representation; CA-to-SA raw
coordinate cosine does not establish fidelity in a common representation space.

Parent reconstruction scores may be reused only after exact endpoint/PNG parity;
they are not newly measured improvements. No natural ground-truth diffusion
trajectory exists for these real artworks. Activation ID, non-promptability and
causal style control remain open regardless of heldout AUC.

## Bounded execution and readback

Proposed maximum2401UNet calls: one inherited discovery plus48 reconstructions
times50 calls, collecting all three representations in the same forward passes.
One A100,8CPU,64GB,60min,no-requeue; no dependent or automatic follow-up jobs.
The initial45min feasibility cap was increased before freezing to allow the
additional pooled reductions and three-representation CPU diagnostics; the
2401-call scientific scope is unchanged.
Actual catalog-based storage bounds and immutable source/runtime/input inventories
must pass before submission. New caches and direction tensors remain remote;
the readback should verify their hashes and consume them remotely when needed.

After publication verification, review reconstructions against originals and
then read feature diagnostics. A later causal experiment starts from separate
existing target images and their own inversions, with each method's own lambda0.
Hold target inversion fixed to compare source directions, or cross source and
target recovery methods to distinguish their effects. Representation, operator,
pooling and source-gallery changes remain separate experimental factors.

Implementation: diff_inversion/modeling/real_representation.py,
diff_inversion/eval/real_representation_runtime.py, scripts/real_representation.py
and scripts/launch_real_representation.py. Receipts will be stored under
reports/technical/real_representation_20261015_v1/.
