# Visible baseline — 2026-09-13

Prepared with prepare_visible_baseline.py; frozen train10825/val1200 hashes and
all label hashes match A. Original visible train JPEGs are reused through links
under /root/autodl-tmp/llvip_visible_seed42_view; no test images or duplicate
image encodings. Labels reference the existing verified infrared label view.

Authoritative source: /root/autodl-tmp/FusionMamba-SACAFM-14e9a93.
Run root: /root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1/full/visible.
Log: /root/autodl-fs/research_protocol/v1/runs/visible_baseline_seed42.log.
Command: python -u research/train_detection_ac.py --mode visible --stage full.

Same YOLO11s weights, seed42, 150 epochs, batch16, img640 and FP32 as A/C/IR.
Initialization is required to match A before training begins. Check live process
and existing artifacts before any launch; never start another copy of this run.
Budget 3.3–3.8 hours, pending observed epoch durations.

After completion check training_complete.json, best weight reload, 150 result
rows and finite metrics. Final extra health callback is best-model validation,
not a 151st training epoch. Add results to WEEKLY_PROGRESS_20260914.md and compare
with IR/A/C. No further architectural experiments are queued by this launch.
