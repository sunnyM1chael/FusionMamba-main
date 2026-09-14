# Image evidence development result — 2026-09-14

CPU analysis completed all 1200 frozen validation images /3778 boxes in162s.
One box had insufficient background pixels;3777 valid paired measurements.
Background excludes all annotated boxes within a15%-of-max-box-side margin.
Pixel values are normalized to[0,1]. These are bounding-box proxies, not object
masks; mean internal absolute gradient is not a boundary sharpness measurement.

For IR-only80 boxes, C-minus-IR absolute foreground/background contrast averages
-0.04212;90% decrease. For C-only52 boxes, the mean is-0.06424;88.46% decrease.
For both-detected3562 valid boxes, mean-0.05329;95.20% decrease.
Internal mean gradient increases by0.00865/0.00784/0.00711 respectively.

Thus contrast reduction and increased local detail are broad fusion effects,
not a discriminator uniquely identifying failures. C-only successes actually
show a larger mean contrast decrease than IR-only failures. Do not infer that
contrast attenuation alone caused the AP gap. Do not equate increased gradient
with restored useful target boundaries; texture/noise may also increase it.

Inspected fixed cases ir_only00/03 and C_only00/03. Fused frames visibly contain
more pavement/structure texture; targets include truncation and overlap in both
directions. These first-manifest cases share a scene and are not representative
of all scenes. Red boxes are ground truth, not displayed detector predictions.
Geometric ghosting and causal alignment failure remain unverified.

Next bounded work: measure boundary-specific gradient with matched interior/
exterior bands, retain foreground texture as a separate measure; add prediction
score/IoU and target size/truncation summaries for both directions and inspect
cases across filename prefixes. Account for repeated scenes before uncertainty
claims. Only choose a candidate after these checks identify a testable mechanism.
No new formal training authorized by the current evidence alone is queued.

Raw data and panels: research/results/image_evidence_v1 locally; server
results/detection_error_analysis_v1/image_evidence_v1. Reproduction:
research/analyze_image_evidence.py. Training artifacts are unchanged.
