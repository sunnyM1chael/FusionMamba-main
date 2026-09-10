# MSRS SACAFM + ACG-AW development result

This is the first full development run, trained with commit `4cfda49` on one
RTX 4090 D. It used the frozen MSRS manifests: 974 training pairs, 109 validation
pairs and 361 test pairs. Training used crop256, batch four, FP32, seed 42 and
100 epochs. Model selection used full-resolution validation fusion loss only.

Validation loss decreased from 2.11808 at epoch 1 to 1.87147 at epoch 74. The
final value was 1.87199. The final ten validation values had mean 1.87215 and
standard deviation 0.00023, indicating a stable plateau. All 24,300 optimizer
updates were applied and no updates were skipped.

The epoch-74 checkpoint generated all 361 test outputs with tiled FP32 inference.
Evaluation used strict filename equality, OpenCV 8-bit grayscale images and
source-average SSIM.

| Metric | Mean | Per-image standard deviation |
|---|---:|---:|
| EN | 6.44875 | 0.83589 |
| SD | 38.31976 | 13.57650 |
| SF | 10.51291 | 3.51755 |
| AG | 3.98132 | 1.62210 |
| MI | 2.54361 | 0.61810 |
| SSIM | 0.71035 | 0.08764 |

Checkpoint SHA-256: `dc5ec1e556150bd6053c27f1b298576828f024734fb7250d6d8dbee4d43b5aa8`.
Test manifest SHA-256: `933b68f7bafa1ca85fd3ad33553a5f6d7275cbc7c2024d79727300a0a210e3f0`.
Metrics JSON SHA-256: `778f84f2f2358809b58845913707d6deefdc0b69fc63da3ba5e4b7f72d86909a`.

These values establish the full model's development baseline. They do not by
themselves establish an improvement over FusionMamba or attribute gains to
SACAFM/ACG-AW. The next controlled comparison should keep SACAFM fixed and train
equal, ordinary learned and ACG-AW weighting under the same protocol. A verified
original FusionMamba baseline is also required before publication claims.
