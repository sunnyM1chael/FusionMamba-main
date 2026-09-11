# SADFFM A/B/C formal development result

Date: 2026-09-11. This is a validation-stage research decision, not a test-set
claim. MSRS test images were not used. All modes use the corrected fusion DSDAM
layout, common exact initialization for same-name/same-shape tensors, equal
modality weighting, seed 42, FP32, crop 256, batch 4 and 100 epochs.

## Completion and integrity

Each run completed 100 epochs with 243 optimizer updates and zero skipped updates
per epoch. Best and last checkpoints for all modes loaded successfully; every
model tensor inspected was finite. Best checkpoints are weights-only and final
checkpoints retain optimizer state.

| Mode | Best validation loss | Best epoch | Last-10 mean | Last-10 population SD |
|---|---:|---:|---:|---:|
| A: DFFM / no spatial enhancer | **1.864409649** | 93 | **1.864850196** | 0.000272898 |
| B: self-conditioned DSDAM + DFFM | 1.867945544 | 89 | 1.868633855 | **0.000200661** |
| C: joint-conditioned SADFFM | 1.866732974 | 91 | 1.867270965 | 0.000256597 |

C is 0.0649% lower than B but 0.1246% higher than A. On clean validation images,
C has lower per-image loss than B on 77/109; paired bootstrap mean C-B is
-0.0012126, 95% interval [-0.0016541, -0.0007830]. Against A, C wins only 22/109;
paired C-A is +0.0023233, interval [+0.0017820, +0.0028985]. Lower is better.
Thus joint conditioning improves the simple DSDAM control but does not recover
the no-spatial baseline's clean-data performance.

Clean best-checkpoint loss components [total, intensity, SSIM-loss, gradient]:

* A: [1.864409649, 0.002127772, 0.179655417, 0.046577767]
* B: [1.867945544, 0.002148542, 0.179926257, 0.047197533]
* C: [1.866732974, 0.002166584, 0.179782337, 0.047243764]

C does not dominate A on any reported loss component.

## Pre-declared synthetic shift analysis

The visible input was shifted right or down by 2/4/8 pixels with replicate
padding. Loss was evaluated against the original aligned source pair after an
8-pixel boundary crop. This is a controlled sensitivity/recovery experiment,
not real unregistered data, registration ground truth, or downstream detection.

| Condition | A | B | C |
|---|---:|---:|---:|
| clean | **1.864410** | 1.867946 | 1.866733 |
| right 2 | **2.091510** | 2.092592 | 2.092576 |
| down 2 | 2.137614 | **2.137097** | 2.137527 |
| right 4 | 2.343209 | **2.342482** | 2.343350 |
| down 4 | 2.446322 | **2.442119** | 2.444948 |
| right 8 | 2.666998 | **2.662496** | 2.664005 |
| down 8 | 2.844422 | **2.836498** | 2.841470 |

C becomes numerically better than A for 4/8-pixel down shifts and both 8-pixel
shifts have paired C-A intervals below zero: right8 -0.0029933
[-0.0044565, -0.0015518], down8 -0.0029519 [-0.0047119, -0.0012302]. However,
B is numerically best for five of six shifted conditions and has larger gains
than C at 4/8 pixels. Therefore this experiment supports some robustness from
spatial enhancement, but does NOT support the joint cross-modal SADFFM design
over the simpler self-conditioned DSDAM control under synthetic shifts.

## Cost and decision

Previously measured 256x256 batch-1 FP32 fusion inference (five warmups, twenty
CUDA repetitions; no I/O/detector): A/B/C 43.06/55.35/57.11 ms; peak allocated
memory 1416.49/1788.16/1792.70 MiB; parameters 320.38/341.27/342.17 M. FLOPs
remain partial because the profiler omits selective scans and deformable ops.

Under the pre-agreed decision rule, C's clean result is worse than A and its
synthetic-shift result is not better than B despite higher cost. Seed 42 alone
does not establish broad generality, but current evidence is insufficient to
retain SADFFM as a demonstrated innovation. Do not spend two more seeds merely
to rescue a negative primary result. Preserve the experiment and report it as
development evidence. Do not rename the same design to manufacture novelty.

Next research action should be a mechanism/design review before further fusion
training. A downstream detector check could answer whether loss misses useful
task information, but it cannot retroactively prove geometric alignment. No
detector or ACG-AW training was launched by this analysis.

Raw analysis: research/results/formal_validation_analysis_v1.json locally and
formal_validation_analysis_v1.json in the server formal-run root. Analysis used
109 frozen validation pairs, deterministic 10,000-draw paired bootstrap seed
20260911, and took approximately 237 seconds of server execution.
