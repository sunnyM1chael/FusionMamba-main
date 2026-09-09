# Research protocol v1 — preparation and release gates

This protocol is a development candidate, not a claim that experiments have
finished. Server data and source code are authoritative. Raw data and previous
results must be preserved; new manifests/results use versioned output folders.

## Scientific questions

1. SACAFM: whether joint alignment/fusion improves correspondence and robustness.
2. ACGAW: whether confidence-sensitive weights improve responses to modality degradation.
3. ESSD: whether DSDAM improves small-object detection independently of P2/capacity.

Separate training is a two-stage pipeline, not end-to-end optimization. Turning
off DSDAM alone must not be described as reproducing original FusionMamba.

## Data roles and partitions

* MSRS: fusion development within published 1,083 train pairs; preserve 361 test.
  Initial development split is illumination-stratified image-level 90/10. It is
  NOT scene-disjoint. Review near duplicates and disclose benchmark limitations.
* LLVIP: preserve 12,025 training / 3,463 test pools. Candidate internal validation
  holds complete 2-digit filename-prefix groups closest to 10% of training.
  Prefix semantics still require confirmation; do not call them verified scenes.
* M3FD: 4,200 detection pairs. Previous 3,141/785/274 is legacy development data,
  not an established benchmark. Locate comparison-method manifests or establish
  and publish a scene/near-duplicate-aware 70/10/20 protocol. Retrain competitors.
* M3FD fusion300: all 300 pairs are pixel-identical to pairs in the 4,200-pair
  detection collection. It is a named subset, not an independent dataset/test domain.
* KAIST: preserve official training/test boundaries and declared annotation,
  sampling and evaluation protocols. Current server has only a subset of sets;
  do not claim a complete benchmark. Existing randomized-all-sequence split is legacy.
* TNO: 37 available pairs, supplemental external evaluation only. Prior smoke
  weights trained on TNO are ineligible for unseen-TNO claims. Disclose any prior
  design feedback from TNO evaluation.

No held-out detection images can train the fusion front end, even without labels.
Images shared by fusion/detection directories count once, not as independent data.
Do not change official test membership to hide leakage; report any issue and use
a separately named sensitivity protocol where appropriate.

## Execution order

1. Decode/hash/annotation audit; review empty/invalid boxes and near duplicates.
2. Candidate manifests, then human/source-based scene/version verification.
3. Strict loader/evaluator tests, subset overfitting and branch-gradient checks.
4. Fusion baseline and ablations on MSRS; choose configurations on validation only.
5. Freeze fusion checkpoint; train VIS, IR and fused LLVIP detectors on identical splits.
6. DSDAM × P2 factorial ablation; fusion × detector factorial comparison.
7. M3FD multi-class confirmation; KAIST extension when benchmark data is complete.
8. Controlled shifts/degradations, cross-domain tests, efficiency and failure cases.
9. At least three seeds for core comparisons; freeze configuration before final tests.

Start budgets: fusion 100 epochs / crop256 / batch4 / lr1e-4; detector150 epochs /
img640 / one fixed YOLO11 scale. These are pilot settings, not guaranteed optimal.
Measure GPU memory/time first. Never silently substitute CPU for an intended GPU run.
Fusion training defaults to FP32: a BF16 smoke run produced a non-finite loss and
FP16 skipped early optimizer updates. Mixed precision requires separate validation.

## Comparison and evaluation

Fusion: original, component ablations, integrated SACAFM, equal weights, ordinary
learned gate, ACGAW. Verify original architecture before calling it a baseline.
Detector: neither DSDAM nor P2, DSDAM only, P2 only, both; isolate further energy
or scale gates. Match initialization, schedule, effective batch, data and resize.

Primary detection metric: AP@[.50:.95]; secondary AP50, per-class AP, small-object
AP/recall with declared coordinate-scale thresholds and instance counts. KAIST
also requires its declared miss-rate evaluation protocol. Report scene-bootstrap
intervals when scene IDs are available and mean/std over seeds. Do not cherry-pick.

Fusion currently implements EN/SD/SF/AG/MI/source-average SSIM. Validate reference
VIF/Qabf implementations before adding them. Noise/sharpening can inflate several
metrics. Missing/extra fused samples must be errors, not silently intersected.
Quality/cost reporting includes both fusion and detector, with explicit batch,
resolution, dtype, warmup and hardware. Color policy must match across methods.

Shift tests specify reference coordinates, resolution, directions and boundary
handling; detection annotations stay attached to the declared reference modality.
Weights/offset visualizations supplement quantitative evidence, not replace it.

## Reproducibility and model usage preference

Store code commit, sample-list hashes, data/annotation provenance, configuration,
seed, environment, checkpoint and per-sample outcomes. Core best checkpoint rules
are fixed before testing. Different loss objectives are not comparable by raw loss.

User preference: Astra for difficult protocol/architecture review; GPT-5.6 Sol for
implementation; GPT-5.6 Luna at max reasoning for bounded routine work. Current
main-turn model cannot be changed through the available tools; no automatic switch
has been performed. Keep progress in this file/reports so later models can resume.

## Current blockers

Server CUDA was restored after preparation. Full CUDA forward/backward tests and
three-epoch FP32 smoke training passed for equal, learned and ACGAW weighting.
Baseline, P2-only, DSDAM-only and full ESSD detectors each completed training,
checkpoint reload and validation on the same LLVIP pilot manifest. The complete
ACGAW fusion-to-ESSD pilot also passed. These pilots validate execution only; their
metrics are not scientific results.

DSDAM uses torchvision DeformConv2d, whose backward input gradient does not have a
strictly deterministic CUDA implementation in the pinned environment. Detection
runs fix their random seed but leave strict deterministic kernels off by default;
core results therefore require repeated seeds and mean/standard-deviation reporting.
The custom detector is constructed once with the dataset class count. Its final
best-checkpoint validation runs in an isolated process to avoid reinitializing the
custom deformable model inside the completed training process.

M3FD group provenance, LLVIP annotation/prefix verification and complete KAIST
benchmark data remain prerequisites for claims on those corresponding protocols,
but do not block MSRS fusion or official-pool LLVIP development training.
