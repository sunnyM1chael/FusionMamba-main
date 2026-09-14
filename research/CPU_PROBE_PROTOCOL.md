# Fixed CPU input probe

Launched 2026-09-14 under the authorized failure-analysis workflow.
Script: run_cpu_input_probe.py. Server output:
results/detection_error_analysis_v1/cpu_input_probe_v1.

24 unique images: first four unique images in sorted filename order for each
of IR-only/C-only/both target groups and each of filename prefixes01/15.
This is an outcome-selected diagnostic subset, not a random validation sample.
Annotations select and score targets, but do not construct input transformations.

Frozen C best detector, CPU one thread, FP32, imgsz640, rect=True, batch1,
prediction confidence0.001/NMS IoU0.7/max_det300. Diagnostic matching uses
confidence0.25/IoU0.5 with confidence-ordered one-to-one matching.

Inputs: original C;0.75C+0.25IR; C transformed to the blended image's global
mean/std and clipped. All round to8bit identically. The latter control tests
whether global intensity statistics can explain an apparent blend benefit;
clipping can prevent exact moment equality, so it is only an approximate control.

No alpha tuning, no retraining, no final-test use. All variants re-predicted in
the same CPU batch1 setting; compare within this probe, not blindly against GPU
batch16 exported counts. Results on selected failure cases cannot establish
full-validation AP improvement. Original baseline target membership may differ
slightly due to inference settings; preserve actual probe baseline measurements.

Finish check: complete.json with72 prediction records,24 images and3 variants.
Report gains and regressions in all three groups. If effects are mixed, do not
launch a full training run as though the mechanism were demonstrated.
