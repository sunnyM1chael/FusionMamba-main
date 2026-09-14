# ESSD detector factorial audit — 2026-09-14

## Experimental question

Use one 2×2 factorial experiment on the same frozen fused M3FD input:

- baseline: standard YOLO11s, P3–P5;
- P2-only: standard backbone, P2–P5;
- DSDAM-only: DSDAM-enhanced backbone, P3–P5;
- ESSD: DSDAM-enhanced backbone, P2–P5.

This separates the main effects of DSDAM and the P2 head and exposes their
interaction. It does not claim parameter equality: DSDAM and P2 necessarily add
capacity. Parameter count, FLOPs and latency must accompany accuracy results.

## Corrections made before formal training

`yolo11_dsdam_deploy/train_essd.py` now sends all four variants through the same
single-build trainer and isolated best-checkpoint validation path. Previously the
baseline used `YOLO.train()` while custom variants injected a prebuilt model.

The formal defaults now select scale `s`, run a fixed 150 epochs with early
stopping disabled, and explicitly freeze AdamW, learning-rate endpoints, weight
decay, warmup, cosine schedule, mosaic closure, AMP, cache, rectangular batching
and plot generation. CLI options remain available, but a formal run manifest must
record the resolved values.

Every run now writes `initialization_audit.json` before optimization. It records
the checkpoint and file hash, seed, resolved arguments, parameter count,
transferred tensor count, a SHA-256 digest computed from source tensor names and
values, and the full initialized model state hash. The
separate `research/audit_essd_initialization.py` preflight builds all variants and
fails unless their common pretrained tensor counts and hashes agree.

The local YOLO11n preflight passed for all four variants: each transferred 378
tensors and produced the same source hash prefix `dec26a2240cc`. The original
project-root `yolo11s.pt` was truncated/corrupt, so it was not used. A clean
YOLO11s checkpoint was downloaded from the official Ultralytics v8.3.0 asset
release to a separate location; its file SHA-256 is
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
The scale-s preflight passed both locally and on the CPU-only server: all
variants transferred 378 common tensors with source-tensor hash prefix
`5b0296b8d606`. The server copy has the same checkpoint file hash. The audit resets seed 42 before
constructing every variant; full-model hashes intentionally differ because the
architectures and their new tensors differ.

## Remaining gates

1. recover or freeze a scene-disjoint M3FD split;
2. materialize one fixed fused-image dataset and confirm exact membership/labels;
3. run one-epoch scale-s smoke tests, compare memory/time, and inspect finite
   losses plus best/last checkpoint reload;
4. only then run formal seed 42. If the effect is competitive, repeat baseline
   and final ESSD with seeds 0 and 1; because deformable-convolution backward is
   not strictly deterministic, report mean and standard deviation.

The full 2×2 should be treated as the detector innovation ablation, not as an
external-method comparison. External fusion methods are evaluated later with the
same detector and manifests after the internal design is frozen.
