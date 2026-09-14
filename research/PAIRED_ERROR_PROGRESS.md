# Paired detector error analysis — 2026-09-14

Completed diagnostic matching on 1200 frozen LLVIP validation images / 3778
person boxes. All four prediction exports share this split. Matching is
confidence-descending greedy one-to-one at IoU0.5/0.75, with confidence
0.1/0.25/0.5. These operating-point counts are diagnostic, not AP, and may
differ from evaluator matching. No official test data or new training used.

At confidence0.25 / IoU0.5: infrared TP3643/FP457/FN135; C TP3615/FP400/FN163.
80 GT boxes are detected only by infrared, 52 only by C, 3563 by both and 83
by neither. Disagreement spans120 images. C has fewer FP but more misses.
At IoU0.75 with the same confidence: IR TP2930, C TP2881, A TP2833.
C improves tight localization counts relative to A but does not surpass IR.
Across all six predefined settings IR has more matched GTs than C. Threshold
calibration can affect these counts; they do not identify an architectural cause.

Next: inspect full-validation foreground/background contrast and boundary
detail, stratified by IR-only/C-only/both/neither detections. Review fixed cases
from both directions. Box-region measures are proxies (not pixel masks or
geometric alignment ground truth); quantify association without claiming cause.
Only after this should a mechanism-based candidate be selected. No redesign or
additional formal training is yet justified by these counts alone.

Server raw evidence: research_protocol/v1/results/detection_error_analysis_v1/
paired_errors.json, summary.json and per-mode predictions.json.
Reproduction: research/analyze_detection_errors.py.
Automatic continuation is active in this task, initially every two hours.
