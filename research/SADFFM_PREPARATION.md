# SADFFM preparation — no formal training tonight

## Fixed question and controls

Does cross-modal-conditioned spatial enhancement BEFORE DFFM help more than
unimodal enhancement alone? SADFFM is the working manuscript name; SACAFM remains
the code class for compatibility. Existing DSDAM/DFFM sources must be credited.

* A: spatial_mode=none, weighting_mode=equal.
* B: spatial_mode=independent, weighting_mode=equal. One shared DSDAM enhancer
  processes the two modalities as a balanced batch; each offset depends on its own
  modality. This does not mean two separately parameterized DSDAMs.
* C: spatial_mode=joint, weighting_mode=equal. Joint IR/VIS/difference/product
  features predict two offset sets; the same enhancer processes both modalities.

Positions are identical across B/C: after each modality's encoder stage and
before cross-modal mixing. Independent encoders remain unchanged. Fused outputs
feed decoder skips; they do not replace next-stage unimodal encoder inputs.

## Interpretation limits fixed before formal evaluation

B/C share the enhancement design but NOT identical offset predictors: B uses
the original self-conditioned offset path (including its down/up sampling), C
uses bounded joint offsets. Parameter count and effective offset range differ.
B vs C tests the integrated design, not cross-modal conditioning in isolation.
If claiming conditioning alone is causal, a matched predictor-input control is
needed. Do not hide this difference or call B/C parameter-matched.

Use fixed MSRS 974/109 development manifests; test361 is not consulted for choices.
Proposed formal settings: FP32, crop256, batch4, 100 epochs, seed42, lr1e-4,
min_lr1e-6, weight_decay1e-4. Check live timing before forecasting all runs.
No restart from legacy-layout weights. Tiny runs are execution tests, not ranking.

## Tonight's checks

* Fusion layout inverse test: global/window1/window2, forward and gradient.
* Known-offset deformable sampling direction, zero-offset identity.
* B output in eval does not depend on the other modality's image.
* Full suite passed 20 tests before the three short runs.
* Three short runs: fixed first16 training and first4 validation pairs, two epochs,
  crop128. No official test data; each run must have eight updates, zero skips,
  loadable checkpoint and correct mode restored for inference.

## Tomorrow's release gates (not automatically launched)

1. Review final short-run checkpoint report; no architecture ranking from it.
2. Finalize initialization policy: equal RNG seed alone does not guarantee all
   common tensors match across architectures; create/verify common initialization
   if claiming it is matched. No claim of matched common tensors yet.
3. Record source hashes, manifests, parameter/compute counts and exact commands.
4. Explicitly launch A/B/C only after these gates; no full-run queue is installed.
5. No ACG-AW re-training, detector training or external benchmarks tonight.

Existing successful synthetic sampling tests establish implementation mechanics,
not that a trained model estimates correct geometric correspondence. Reliability
response and detector utility remain separate untested research questions.

## End-of-evening verified status

All three short runs finished two epochs with eight updates and zero skips each.
All three best checkpoints reloaded through the real generation loader, restored
the correct spatial class and passed a CUDA 128x128 finite-output check. Last
checkpoints also loaded with optimizer state and finite model tensors.

| Mode | Total parameters | Short-run epoch seconds |
|---|---:|---|
| A none | 320,381,401 | 42.6 / 40.8 |
| B independent | 341,269,393 | 43.4 / 43.0 |
| C joint | 342,174,217 | 44.5 / 44.2 |

Times include large checkpoint writes on tiny runs and must not be extrapolated
as full-data epoch timings. All groups still instantiate unused equal-gate
parameters; these totals are not active-compute counts.
git diff --check passed (only line-ending notices). Server data disk approximately
110 GB available. No train.py or training queue process remains; GPU showed
0 percent utilization / 1 MiB at final check. No formal training scheduled.
Common-tensor initialization verification and compute profiling remain release
gates before tomorrow's formal start, not claimed completed tonight.

## 2026-09-11 release and launch

User authorized verification and formal training today. Server suite passed 20
tests again. research/spatial_formal.py prepared CPU-only untrained checkpoints:
A supplies common core tensors; B supplies enhancer tensors shared with C; new
tensors retain seed42 defaults. Buffers included; saved files reloaded and compared.
B: 1,578 shared tensors verified, 348 raw-seed mismatches corrected.
C: 1,646 shared tensors verified, 396 raw-seed mismatches corrected.
This matches common initialization, not every subsequent stochastic operation.
Fresh optimizers; no legacy trained checkpoint used.

GPU profiling completed: 256x256 pair, batch1 FP32, TF32 off, five warmups and
20 measured repetitions, no image IO or detector. A/B/C mean CUDA times are
43.06 / 55.35 / 57.11 ms; peak allocated memory 1416.49 / 1788.16 / 1792.70 MiB.
FLOPs are partial: profiler explicitly omits selective scans, deformable operations
and others. Do not report these as full-network FLOPs. Uses tiny-run checkpoints.

Formal serial queue started around 12:05 Asia/Shanghai, A training PID 2429.
Run directory: /root/autodl-fs/research_protocol/v1/runs/spatial_formal_corrected_seed42_v1.
Common initialization file hashes and code hashes recorded in preparation.json;
each run has exact command/provenance. Queue stops on a failed child or invalid
epoch/update counts and checks source hashes before starting each group.
The launch action is now authorized; the earlier no-training restriction applied
to the previous evening. No ACG-AW or detector training has been launched.

Formal queue completed on 2026-09-11 around 21:10. Final conclusions and the
pre-declared validation/shift analysis are recorded in SADFFM_FORMAL_RESULT.md.
