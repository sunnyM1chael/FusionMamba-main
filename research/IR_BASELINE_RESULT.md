# Infrared baseline result — 2026-09-13

Completed all 150 epochs on frozen LLVIP development train10825/val1200,
YOLO11s, seed42, batch16, img640, FP32 and the same initialization/settings as
A/C. Official test was not used. Best model reload/finite-tensor check passed.
results.csv has 150 training rows; health record151 is final best validation.

| Input | Best validation AP50:95 | AP50 | Precision | Recall |
|---|---:|---:|---:|---:|
| A fusion | 0.616212 | 0.970517 | 0.944272 | 0.925357 |
| C SADFFM fusion | 0.621705 | 0.970082 | 0.941787 | 0.937803 |
| Infrared only | 0.636254 | 0.970796 | 0.943643 | 0.939583 |

Infrared best epoch52. Final best-checkpoint validation gives infrared a
1.4549 percentage-point AP50:95 advantage over C and 2.0042 over A. This is
single-seed development evidence: C improves A, but fusion necessity is not
demonstrated against direct infrared detection. Prioritize examining information
loss and failure cases before spending on additional A/C detector seeds.
LLVIP validation has too few standard small objects to substantiate that claim.

Best checkpoint SHA256:
ae2a81d76b4602fdeea69b199929ad23817c51169b35a0b27a233585e42a47cd.
Local backup: research/results/infrared_seed42_final (best.pt, training_complete,
epoch health, results.csv, args.yaml). Large/raw artifacts remain git-ignored.
Server artifacts remain under the full/infrared run documented in IR_BASELINE_PLAN.md.
Original infrared JPEGs were decoded directly; fusion images were lossless PNGs.
No visible baseline or additional seed was launched tonight.
