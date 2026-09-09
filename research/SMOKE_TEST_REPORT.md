# Protocol v1 smoke-test report

Date: 2026-09-09. Hardware: one NVIDIA RTX 4090 D. Smoke tests establish that
the controlled experiment paths execute correctly; their tiny-sample metrics must
not be reported as model performance.

## Fusion

All runs used the same 16-pair MSRS training manifest, four-pair validation
manifest, seed 42, crop 128, batch four and FP32 for three epochs.

| Weighting mode | Train loss, epoch 3 | Validation loss, epoch 3 | Result |
|---|---:|---:|---|
| Equal 0.5/0.5 | 3.93315 | 4.20127 | passed |
| Ordinary learned gate | 3.93097 | 4.19972 | passed |
| ACG-AW | 3.93168 | 4.19546 | passed |

Full CUDA forward/backward and alignment-gradient checks also passed for all three
modes. Equal weighting correctly has no trainable weighting gradient. FP16 skipped
early optimizer steps and BF16 later produced a non-finite loss, so formal fusion
training and generation default to FP32.

## Detection

The four factorial variants used the same LLVIP infrared 16-image training and
four-image validation manifests, seed 42, image size 320 and one epoch. Each run
trained, saved `best.pt`, reloaded it and completed final validation.

| Variant | DSDAM | P2 head | Fused-model parameters | Result |
|---|---:|---:|---:|---|
| YOLO11n baseline | no | no | 2,582,347 | passed |
| P2-only | no | yes | 2,658,868 | passed |
| DSDAM-only | yes | no | 3,588,260 | passed |
| ESSD | yes | yes | 3,664,781 | passed |

The custom variants remapped 378 compatible YOLO11n tensors. They are built once
with the dataset class count (`nc=1` for LLVIP), preventing a second DSDAM stride
probe from discarding or destabilizing the remapped initialization.

## End-to-end chain

The ACG-AW fusion pilot checkpoint generated 16 LLVIP training and four validation
images with tiled FP32 inference and finite-output checks. A strict YOLO view linked
the corresponding derived labels and rejected any missing or extra image. Full ESSD
then completed training, checkpoint selection, reload and validation on those fused
images. This validates the two-stage Fusion-to-Detection data path.
