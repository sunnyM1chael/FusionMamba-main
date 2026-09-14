# M3FD formal detection protocol decision — 2026-09-14

## Decision

Do not start a formal M3FD training run from the locally held legacy split or
from a random image-level split. The current M3FD copy has 4,200 matched pairs
but no split metadata. The consecutive capture structure and unresolved
near-duplicate grouping make image-level random sampling vulnerable to leakage.

First preference is to recover the `meta` directory distributed with the M3FD
data, especially `scenario.json` and its generated `train.txt`/`val.txt`. The
official TarDAL code at commit `6a9edd744b44fc03344fe8fb0fd930f5df47b00b`
uses scenario and scene ranges to generate membership. The official repository
documents the expected meta files but does not itself version those data files.
Therefore a split cannot be called official unless its manifest is recovered
from the original package and its provenance and SHA-256 hashes are recorded.

The inspected fork at commit `c6079fbcf0245ed0db2e8ceb48dadac38f9d0d42`
is not a substitute: `meta_get_TYL.py` samples 2,100 image names with Python
`random.sample`, supplies no seed and assigns the remainder to validation. It is
neither reproducible nor scene-independent.

A local CPU screen decoded all 4,200 pairs and ranked adjacent transitions by
mean IR/visible thumbnail change. Visual review of the top 80 confirms that the
ranking contains both real scene changes and ordinary large camera/subject
motion, so it cannot automatically define groups. It also found 58 groups with
equal combined IR/visible dHash; these must remain within one partition pending
full-resolution confirmation. The machine-readable screen is
`research/results/m3fd_sequence_screen_local.json`, with four contact sheets in
`research/artifacts/m3fd_boundary_contacts` (kept as local review artifacts).

Manual review has started with the 20 highest-scoring transitions. The review
record is `research/results/m3fd_boundary_review_top20_v1.csv`: 9 are marked
high-confidence boundaries, 5 are compatible with same-scene motion, and 6
remain uncertain because a moving-camera sequence can cross tunnels, roads, or
nearby viewpoints. These labels are triage evidence only; uncertain transitions
must not be used to freeze groups without adjacent context.

Ranks 21--40 have also been reviewed in
`research/results/m3fd_boundary_review_21_40_v1.csv`: 7 are high-confidence
boundaries, 6 are compatible with same-scene motion, and 7 remain uncertain.
Across ranks 1--40 the running totals are therefore 16/11/13. No group manifest
is frozen from these transition labels alone.

## Frozen fallback if original metadata cannot be recovered

Create and explicitly name a **custom scene-disjoint protocol**, never an
official M3FD split. It must satisfy all of the following before training:

1. infer candidate sequence boundaries from consecutive IR and visible frames,
   then manually verify the complete sequence using contact sheets (the current
   top-80 screen is triage, not a finished grouping);
2. place an entire scene and every exact/equal-dHash group in one partition;
3. target 70/10/20 train/validation/test membership (approximately
   2,940/420/840 images), allowing count deviations to preserve whole groups;
4. balance day/night and the six detection classes at scene level and publish
   the achieved counts rather than silently moving individual frames;
5. write sorted `train.txt`, `val.txt`, and `test.txt`, plus a JSON audit with
   dataset root, generation version, seed (if tie-breaking is required), class
   counts, small-object counts, scene IDs and SHA-256 hashes;
6. freeze the test manifest before model development. Use only train/validation
   for module choices; run the test set once for the final selected models.

This protocol is named `M3FD-SG-70/10/20-v1`. Seed 42 may only break ties in a
group-level assignment; it must not shuffle individual frames. The 20% test set
supports more stable rare-class estimates, while 10% validation is sufficient
for the deliberately limited model-selection stage. These ratios are our design
choice, not evidence of membership used by an M3FD paper.

## Comparison rule

All internal factorial groups and reproduced external methods must use these
same frozen manifests. Published numbers based on unknown or different M3FD
splits may be quoted only as contextual results, not placed in the same direct
comparison column. The paper must report the manifest files or a permanent
repository link so reviewers can reproduce the comparison.

## Training gate

Formal M3FD training remains blocked only by split provenance/scene grouping and
the fused-image materialization for the frozen membership. Dataset integrity and
small-object volume have already passed. A short engineering smoke test may run
before the split is frozen, but its metrics must not enter the paper.
