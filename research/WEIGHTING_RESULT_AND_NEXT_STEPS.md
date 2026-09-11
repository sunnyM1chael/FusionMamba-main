# MSRS weighting controls: completed development comparison

Date: 2026-09-10, approximately 21:10 Asia/Shanghai.

## Verified results

All three runs finished 100 epochs, with 24,300 recorded optimizer updates and
zero skipped updates per run. All validation totals are finite. Six best/last
checkpoints loaded on CPU; model tensors are finite. Best checkpoints are
weights-only and last checkpoints contain optimizer state. This is a load and
model-tensor check, not a resumed training or inference equivalence test.

| Mode | Best validation loss | Best epoch | Last-10 mean | Last-10 population SD |
|---|---:|---:|---:|---:|
| Equal | 1.872236382 | 74 | 1.872785116 | 0.000216978 |
| Learned | 1.870807475 | 74 | 1.871635225 | 0.000242775 |
| ACG-AW | 1.871465321 | 74 | 1.872149295 | 0.000228933 |

Lower is better. These are fusion losses, NOT detection AP. Epoch-to-epoch SD
is not independent-seed uncertainty and cannot establish significance.

Manifest hashes, model/loss/data-source hashes and main training settings match.
ACG-AW was resumed after a checkpoint-saving change; its initial provenance has
an older train.py hash. The reviewed checkpoint-fix diff changes saving/resume
handling, not the loss formula. This interruption remains a disclosed limitation.
Only seed 42 has been evaluated. Test images were not evaluated in this analysis;
the earlier ACG-AW test evaluation remains part of the project history.

Loss = 10 * SSIM-loss + 10 * intensity-MSE + gradient-L1.
Last-10 component means:

| Mode | Intensity | SSIM-loss | Gradient |
|---|---:|---:|---:|
| Equal | 0.001869631 | 0.180784970 | 0.046239098 |
| Learned | 0.002052963 | 0.180483337 | 0.046272224 |
| ACG-AW | 0.001922328 | 0.180662790 | 0.046298108 |

Learned has the lowest aggregate loss but does not dominate every component.
ACG-AW has no demonstrated advantage over Learned under this development setup.
Neither superiority across seeds nor downstream detection benefit is established.

## Decision and ordered experiment checklist

1. **Next, no retraining:** evaluate best checkpoints on the SAME frozen 109-image
   MSRS validation set. Record per-image losses, fixed qualitative examples,
   modality-weight variation, temperature, saliency gain and alignment confidence.
   Check collapsed/near-equal gates and residual-path dominance. These are possible
   explanations, not established causes. Actual image/weight diagnostics are pending.
2. **Weighting decision:** keep Equal as the minimal reference, Learned as the
   current numerical candidate, and ACG-AW as unverified. If guided weighting has
   no meaningful validation benefit, do not force it into the novelty claims.
   Do not change losses merely to make the preferred module win.
3. **Fusion screening:** with one provisional weighting choice, compare the
   spatial-module baseline and integrated design. Turning off DSDAM alone is not
   an original FusionMamba reproduction. Verify architecture before naming it so.
4. **Small downstream development check:** only baseline and promising fusion
   candidate(s), same YOLO11 architecture/training protocol on LLVIP training and
   internal validation. Audit scene/prefix and leakage issues before execution.
   Use validation, not official test, for choices. Loss ties cannot resolve AP.
5. **Detector ablations:** freeze provisional fusion, distinguish DSDAM and P2
   effects if P2 is retained. Define small-object evaluation in advance.
6. **Freeze final design**, then run external representative-method comparisons,
   final-configuration ablations and core seed replications. Prioritize detection
   evidence in the single fusion-to-detection chapter; fusion quality supports it.
   Confirm M3FD split before secondary detection experiments. Incomplete KAIST is
   not a prerequisite. Do not train every fusion x detector combination.

Existing three checkpoints/results remain reusable when the protocol is unchanged.
Changing our fusion output requires regenerating that output and retraining its
detector. Competitor runs remain reusable if their data/evaluation protocol stays
unchanged. No new training has been launched. Training polling can stop now.

## Reproduction

Run research/summarize_weighting_runs.py on the server via stdin with
--verify --compact to reproduce summaries and checkpoint load checks.
Server run root: /root/autodl-fs/research_protocol/v1/runs.

## Completed inference diagnostic (same day)

research/diagnose_weighting.py evaluated all 109 frozen validation pairs using
full-resolution FP32 and strict best-checkpoint loading. All three mean losses
exactly reproduce the recorded best values. Server diagnostic execution took
39.1 seconds (excludes implementation, transfer and interpretation).

Across 109 images and four gates, averaging each gate/image statistic equally:

| Mode | Mean IR weight | Mean absolute departure from 0.5 | Fraction within 0.01 of 0.5 |
|---|---:|---:|---:|
| Learned | 0.47067 | 0.09905 | 0.23523 |
| ACG-AW | 0.48855 | 0.15426 | 0.10743 |

For the first eight fixed manifest images, replacing all gates with equal weights
at inference changes output by mean absolute 0.04314 (Learned), 0.02864 (ACG-AW),
in normalized image units. These are off-training-distribution interventions,
not quality improvements or retrained ablations. Gates are not simply inactive.
ACG-AW has lower per-image loss than Learned on 41/109 validation images.
The four ACG-AW saliency gains are approximately 0.181, 0.144, 0.129, 0.093.
Neither constant equal weights nor a completely untrained saliency gain explains
the near tie. This does not establish all gradients are currently healthy.

Visual inspection of fixed image 00 shows vehicle ghosting in all three variants.
This is a case-level failure, not a measured dataset-wide ghosting rate. Prioritize
checking whether spatial alignment actually reduces known correspondence errors.
Feature-distance confidence is a heuristic, not calibrated geometric correctness.

Artifacts: research/results/weighting_diagnostic_20260910_v1 (raw JSON and 24
IR/VIS/fused panels); server same name under protocol/v1/results. No new training.

## Time budgets and stop gates

These are planning ranges, not guaranteed completion times; detector timings need
a representative pilot. One GPU, serial execution assumed.

* Completed weight diagnostic: 39.1 seconds server inference, several minutes setup.
* Alignment code/path audit plus fixed validation-case checks: 30-60 minutes.
  No architectural change merely to reach a target novelty count.
* If a specific correctable cause is found: minimal implementation and analytic /
  synthetic known-shift tests, 1-3 hours. Stop if mechanism tests fail.
* Training-data-only small pilot for a justified change: budget 20-40 minutes per
  candidate; do not claim final convergence from a pilot. At most one candidate
  initially; stop for nonfinite/ineffective paths or lack of the intended behavior.
* A retained fusion configuration: approximately 3-3.5 hours per 100-epoch run
  based on existing measurements. Two runs roughly 6-7 hours, only after screening.
* Downstream setup and representative timing pilot: 30-60 minutes; extrapolate
  formal detector runs from 3-5 representative epochs including validation/save.
* External benchmark reproduction budget: unknown until methods and protocols are
  chosen; do not promise a fixed aggregate runtime or launch all combinations.

No experiment-free guarantee of scientific utility is possible. A new named
module is not automatically an innovation. Prefer two substantiated contributions
to three unverified ones; novelty review and final controlled validation remain
necessary even when cheap screening succeeds.
