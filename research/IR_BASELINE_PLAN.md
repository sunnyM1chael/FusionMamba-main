# Infrared baseline — 2026-09-12 evening

User authorized the infrared-only YOLO11s experiment tonight. The preparation
checks frozen LLVIP train10825/val1200 membership and exact label hashes against
the completed A export report, then starts 150 epochs through the same trainer.
YOLO11s weights, seed42, initialization, batch16, imgsz640, FP32 and all training
options match A/C. The original infrared JPEG view is reused without duplicating
images/labels or reading the official test set. Fused A/C inputs were PNGs;
infrared inputs retain the original JPEG decoding (no additional lossy encoding).

Server source: /root/autodl-tmp/FusionMamba-SACAFM-14e9a93.
Run: /root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1/full/infrared.
Log: /root/autodl-fs/research_protocol/v1/runs/infrared_baseline_seed42.log.
Preparation script: research/prepare_ir_baseline.py.
Training command: python research/train_detection_ac.py --mode infrared --stage full.
Both have already been queued serially with failure stopping the next command.
Do not relaunch without checking processes and artifacts.

Initial model hash must equal A/C:
12b4ecc3e5eff61eaf987cd0ee045b68adda7c7b05d23ad5d24f8fad3bd87aff.
Estimate 3.3–3.7 hours after training starts; refine after 3–5 real epochs.
Final extra epoch151 health callback is best validation, not another train epoch.
On completion verify weights and all150 CSV rows, compare best AP50:95/AP50/P/R
with A/C, preserve single-seed caveat, write report, commit and stop the monitor.
No visible-only or extra-seed experiment is queued for tonight.
