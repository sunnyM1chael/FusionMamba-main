# Mechanism-first audit, 2026-09-10

No architecture or server checkpoint changed in this audit. No new training.

## Intended hypotheses and placement

| Component | Prior rationale | Actual placement | Limits / falsification |
|---|---|---|---|
| Cross-modal deformable enhancement | Cross-modal correspondence errors should be addressed before location-wise feature mixing | Each encoder stage, after unimodal blocks, before DFFM | Deformable sampling does not itself establish geometric alignment; require known-correspondence tests |
| Modality gate | Relative usefulness varies spatially and by feature channel | Before DFFM differential interaction, also contributes a residual fusion output | Feature contrast is not necessarily target utility; equal-gate baseline is necessary |
| Confidence guide | Uncertain correspondence should reduce strong mixing preferences | Scales both modality logits before softmax | exp(-mean absolute feature difference) is scale-sensitive and is not calibrated geometric confidence; independent encoders need not share channel semantics |
| Detector DSDAM | Preserve local spatial evidence before repeated downsampling | Inspected DSDAM-only YAML: after C3k2 at strides 4, 8, 16 | Three placements are a design choice, not proven necessary; enhancement cannot recover all detail already removed by stride |

Fused features enter the decoder via per-scale outputs. Next encoder stages receive
the original unimodal feature streams after downsampling, not the aligned/fused
outputs. Consequently this is not progressive cross-stage geometric registration.
The module positions are structurally motivated, but neither optimal placement nor
independent novelty follows from the code comments.

## Confirmed implementation discrepancy

Server DSDAM.py restores attention [B, heads, H*W, head_dim] using
permute(0,2,1,3).view(B,C,H,W), interleaving pixel and channel identities.
The required inverse arrangement is permute(0,1,3,2).contiguous().view(B,C,H,W).
Read-only analytical probe research/audit_fusion_attention_layout.py injects a
known attention output into the actual deployed module and captures out_proj's
input. Against an explicit per-index reference, 22/24 entries mismatch. The
correct permutation matches all entries. This is not inferred from validation loss.

The prior attention correction covered the detector implementation, not this
fusion implementation. All three completed fusion runs share this discrepancy.
Their measurements remain valid descriptions of that implementation; they are
not sufficient evidence against the intended correctly arranged mechanism.
Do not present this discrepancy as the proven cause of all ghosting or near ties.

## Bounded next gate

Before any training: implement a versioned minimal correction and regression tests
for both window/global attention, run forward/backward checks (budget 15-30 min).
Preserve old checkpoints and label old results; never silently overwrite the old
implementation and evaluate old weights as if they were corrected-model training.
Then at most one 20-40 minute development pilot to establish trainability, not
final superiority. Decide on a matched full baseline only after those checks.
Avoid automatically rerunning all three 100-epoch experiments.

Novelty cannot be guaranteed by a mechanism argument. Define hypotheses before
new evaluations, distinguish debugging from method selection, and do not add a
third module solely to reach a contribution count.

## Minimal fix completed

Fusion DSDAM layout corrected locally and on the server. Original server source
preserved as DSDAM.pre_fusion_layout_fix_20260910.py, SHA256
1b8ce862ea93fe8667fd0ea6c621da4fb30876d61a6c5c7bad7617e0271a73ae.
Old checkpoints untouched; they belong to the legacy-layout implementation.
Do not run them under corrected code and call the results a fair corrected-model
comparison, nor silently resume them as the original experiment.

New tests/test_fusion_layout.py failed all three cases before the fix and passed
after it (global, window=1, window=2, including analytical input gradients).
Full server suite: 18 passed. Full CUDA FP32 128x128 forward/backward smoke check
passed; offset-head and adaptive-weight gradients were nonzero. No new training.

Research status: legacy clean-validation weighting comparison completed;
response to controlled modality reliability changes NOT tested; causal explanation
linking alignment quality to weighting performance NOT established. Next spatial
module hypothesis must hold weighting fixed to avoid confounding. User calls the
integrated DSDAM+DFFM idea AMDFFM; code currently calls it SACAFM. Confirm naming
in the write-up rather than silently treating differently named methods as distinct.
