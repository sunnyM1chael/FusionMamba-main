# A/C downstream development decision — 2026-09-11

UPDATE 2026-09-12: proceed with official YOLO11s as the primary detector. The
existing local yolo11s.pt was detected as a truncated ZIP (1,471,531 bytes) and
must never be used. Server file yolo11s_official.pt was downloaded from the
Ultralytics v8.4.0 release, loads as 9,458,752 parameters, and has SHA256
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5.

PAUSED 2026-09-11: user requested that formal work continue tomorrow. The
detached preparation queue was terminated safely during A full export. A has
3,785/10,825 training PNG files present; C full export has not started. Formal
detector training has not started and the monitoring heartbeat is paused. Before
resuming, decode/verify the last partial output and make the exporter skip only
verified existing files. Confirm YOLO11s with the user before formal training.

RESUMED 2026-09-12: exporter now validates every existing PNG's decode, shape,
dtype and copied-label hash before reuse; new images use atomic partial-file
replacement. A/C export resumed. Formal YOLO11s remains release-gated.

INODE RECOVERY 2026-09-12 12:49 CST: C export stopped at 10,055/10,825 train
images because the 200,000-file data-disk inode quota was full (78 GB remained).
No formal training had started. The redundant, reproducible LLVIP visible YOLO
view was archived before removal at
llvip_yolo/visible_derived_archive_20260912.tar: 30,990 archive entries, SHA256
3f5ee038519b87826b93c559eadb1523ddc759b84931e27667758aea4b7736ec.
The raw LLVIP dataset and infrared YOLO view were not changed. This released
30,990 inodes (85% used), and the verified-resume exporter was relaunched. The
archive is recoverable with tar if that derived visible-only view is needed.

Before detector results, the frozen validation size audit counted 3,778 boxes:
only 2 have native area <32^2, and only 13 have resized-to-640 area <32^2. Thus
LLVIP can support the overall fusion-to-person-detection decision but cannot
provide a credible standalone standard small-object AP claim. Do not derive a
new threshold from observed detector results. DSDAM small-target evidence needs
a dataset/split with an adequate predeclared small-object count.

User approved proceeding with the reviewer plan. Goal: determine whether corrected
A/C frozen fusion checkpoints preserve information useful to a standard detector.
Do not claim that AP alone proves geometric alignment. Do not add detector DSDAM,
P2, a new fusion module, or ACG-AW runs to this comparison.

Server source: /root/autodl-tmp/FusionMamba-SACAFM-14e9a93.
Run root: /root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1.
Preparation log: /root/autodl-fs/research_protocol/v1/runs/detection_ac_preparation_v1.log.

## Completed gates

- MSRS matched-crop analysis v2 completed without retraining; see spatial report.
- LLVIP frozen development: 10,825 training pairs / 1,200 validation pairs,
  both from official training. Official 3,463 test pairs untouched by this job.
- Historical decoded-pixel hashes and dHash checked across this internal split
  separately for both modalities: no cross-split candidates. This is not a full
  near-duplicate audit and prefix semantics remain unverified, not scene-disjoint.
- XML dimensions match historical image audit. Shared derived labels retain
  negatives and exclude five previously documented zero-width training boxes.
- A/C strict checkpoint load and four fixed training-image native-resolution
  FP32 exports passed, finite outputs and zero clipping in those eight outputs.
- Native 1280x1024, grayscale PNG; no tiling, rescaling, per-image normalization
  or visible chroma reintroduction. Detector loader expands grayscale to 3 channels.
  This isolates learned fused luminance, not the prior colorized pilot pipeline.
- Three-epoch standard YOLO11n smoke for each mode passed (16 train / 4 val),
  including final best validation, checkpoint reload and finite tensors.
- Detector initial model SHA identical across smoke A/C:
  7234f3ebe2f2a665662460707b1d0796b764ed8c9c5bb6f0a79ad425e5484a8b.
  Smoke AP is execution evidence only, never candidate selection.

## In progress and next release gates

prepare_detection_ac.py is a serial detached queue: smoke exports -> both smoke
detectors -> both full exports -> READY_FOR_FORMAL_REVIEW. Any subprocess failure
stops the queue. The queue does NOT launch full detector training.

Full exports use exclusive new directories, file hashes and shared label copies;
each source filename maps to its same-stem lossless PNG. Estimated total export
time 1.5–2.5 hours and ~18 GB from four native-resolution training pairs per mode;
refine from progress.json. Source data and previous results are not overwritten.

Before formal launch, verify both export reports (12,025 rows each, exact split
membership, matching labels/dimensions, finite outputs), source hashes and no
failure.json. Inspect fixed sample outputs; do not choose samples by performance.
Freeze evaluation spec: primary validation AP50:95, secondary AP50 and recall.
For size-specific evaluation, define native-image box area <32^2 pixels as the
small-object group, report its count and explicitly distinguish it from resized
640-coordinate small-object definitions. Implement/test evaluator before claims;
if native small count is inadequate, disclose rather than tune the threshold on AP.

Then launch standard YOLO11n A and C serially, same pretrained file, seed42,
150 epochs, batch16, imgsz640, AdamW lr0=.001, lrf=.01, cosine, warmup3,
weight_decay=.0005, no early stop (patience0), close_mosaic10, FP32,
deterministic=True. train_detection_ac.py enforces exact initial tensor hash,
fails for nonfinite gradients/loss, and disables automatic batch reduction.
Use new full runs, not smoke resumes. Estimate runtime after 3–5 full-data epochs,
including validation/save, and adjust monitoring to the estimated completion.
Do not infer convergence or utility from those first epochs.

Full training is authorized within the A/C plan, but launch only after these
checks. No all-method comparison, additional seeds or weighting reruns are queued.
At completion compare validation AP and cost, preserve failed hypotheses, then
decide whether further seed replication is warranted. No universal loss/AP
percentage threshold establishes innovation. Official test remains reserved.
