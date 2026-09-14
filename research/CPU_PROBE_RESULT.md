# CPU intervention result — 2026-09-14

Completed24 images ×3 variants =72 predictions in233.78s on CPU, one thread,
0.5-core/2GB instance. No GPU, no training. Protocol: CPU_PROBE_PROTOCOL.md.

Selected IR-only group8 targets: original C1 hit, IR blend3 hits, moment control1.
Mean target score at IoU>=0.5:0.21155/0.37919/0.21354 respectively.
Selected C-only8: hits5/6/5; scores0.59732/0.46935/0.59678.
Both-detected8: hits8/8/8; scores0.86541/0.86378/0.86468.
For IR-only, average best-IoU changes0.68489→0.67086 with blending: score gain
does not establish improved localization. C-only mean score falls despite one
more matched target. Do not describe uniform improvement or quote AP gains.

Baseline CPU single-image matching does not perfectly reproduce the previous
GPU batch16 groups (IR-only now1/8; C-only5/8). These group names refer to
previous exports; all comparisons above use freshly predicted identical CPU
settings within this probe. Preprocessing/batch/precision pathway differences
must be resolved before treating categorical recovery counts as robust.

Conclusion: fixed spatial IR addition shows an exploratory score benefit on
some failures beyond approximate global moment adjustment, but evidence is
mixed, outcome-selected and small. It is not a validated architecture or cause.
Do not start full training. Next first inspect validation-vs-predict resize and
padding settings causing membership differences, then use matched preprocessing
for a larger fixed diagnostic subset (same alpha, same controls, no tuning).

Raw server output: results/detection_error_analysis_v1/cpu_input_probe_v1/
protocol.json,records.jsonl,summary.json,complete.json. Rounding/clipping means
moment control is approximate; next audit should measure achieved moments.

## Matched-canvas rerun

The formal validator used rect=false and a640x640 canvas; v1 single-image
predict used rect=true and512x640. V2 repeated the same24×3 probe with
rect=false/640x640. IR-only selected targets were1/2/2 hits for original/blend/
moment control, with mean score0.20321/0.30497/0.20379. C-only were7/7/7 and
mean score0.61029/0.59156/0.60873. Both were8/8/8 and essentially unchanged.

The apparent extra IR-blend hit is also produced by the moment control, while
C-only localization/score does not improve. V2 therefore does not establish a
specific benefit from spatial IR retention. The no-full-training stop gate fires.

Remaining category mismatch versus the previous GPU export is expected from
CPU/GPU convolution/NMS differences near thresholds and from diagnostic greedy
matching versus the validator's IoU-sorted matching. Single-class multi-label
NMS is not the main issue. Future CPU probes report within-path continuous score/
IoU changes, not recovery of GPU-defined categories. Strict category/AP replay
requires the GPU validator and matched dataloader; it is not warranted by this
mixed small-sample result.
