# Corrected A/C downstream detection result

Date: 2026-09-12. Both configurations completed 150 YOLO11s epochs on the same
frozen LLVIP development split (10,825 train / 1,200 validation), with identical
detector initialization, seed 42, batch 16, image size 640 and FP32. The official
3,463-image LLVIP test pool was not used. A uses DFFM without a spatial enhancer;
C uses joint-conditioned SADFFM. Both fusion checkpoints were selected on MSRS
development validation before LLVIP detector training.

## Completion and integrity

Both results.csv files contain one header plus 150 training rows. Each best
checkpoint reloaded successfully with finite model tensors. Initial detector
state SHA256 was identical:
12b4ecc3e5eff61eaf987cd0ee045b68adda7c7b05d23ad5d24f8fad3bd87aff.

| Mode | Best epoch | AP50:95 | AP50 | Precision | Recall | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|---:|
| A | 68 | 0.616212 | **0.970517** | **0.944272** | 0.925357 | **78.785** |
| C | 78 | **0.621705** | 0.970082 | 0.941787 | **0.937803** | 79.190 |

C improves AP50:95 by 0.005493 absolute (0.549 percentage points, 0.891% relative)
and recall by 0.012446 absolute. AP50 is 0.000434 lower and precision is 0.002486
lower. Detector epoch time is 0.405 seconds (0.51%) higher; this does not include
the separately measured fusion-front-end cost.

The last ten training-epoch validation AP50:95 values have mean/SD
A 0.600029/0.000693 and C 0.609630/0.001063. The final extra health record is an
isolated validation of the selected best checkpoint (labelled epoch 151 by the
callback), not a 151st training epoch.

Best checkpoint SHA256:

- A: c188b615e6a160c925c849ddfa346d0d00e7a79465c47e9d1b0856caac9754b3
- C: 29df27efb7215a47e9f4fb4f3a672ffd500b3e72da440fb81ec75091f025a6e1

## Reviewer-facing decision

This single-seed result changes SADFFM from an unsupported fusion-loss candidate
to a promising downstream candidate: C is worse than A on clean MSRS fusion loss
but modestly better on the primary LLVIP detection metric and recall. It supports
the narrower hypothesis that SADFFM may retain task-useful person information; it
does not demonstrate geometric alignment or general superiority.

Do not use the official test pool or begin external-method comparisons yet. The
next decision experiment is two additional matched A/C detector seeds. Retain the
module as a claimed contribution only if the AP50:95 direction is reproducible and
the mean gain is defensible against the full fusion-plus-detection cost. The
current LLVIP validation has only 13 boxes meeting the predeclared resized-640
small-object definition, so this experiment cannot substantiate a small-target
claim. Detection-side DSDAM needs a dataset with adequate small-object counts.
