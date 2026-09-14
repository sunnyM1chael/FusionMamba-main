# M3FD detection audit progress — 2026-09-14

Read-only inventory of server M3FD_Detection completed before any split or
training decision. Annotation/IR/VIS counts are4200/4200/4200 with4200 matched
stems and no missing modality/annotation. XML contains34,407 valid boxes, no
empty images and no invalid box under the declared XML sizes.

Classes: People11,477; Car18,296; Lamp2,405; Bus700; Motorcycle521; Truck1,008.
After aspect-preserving long-side resize to640,20,364 boxes have area below32²
and10,860 below16². People has8,450 below32² and5,139 below16². These are
planning proxies, not official M3FD metrics. They show adequate scale volume for
a detection-side small-object ablation, unlike the current LLVIP validation.

Header audit also completed all4200 triples: IR/VIS/XML dimensions match for
every stem, with no mismatches. Both modalities are PNG files stored in RGB mode.
Image-size counts match XML inventory;3926 pairs are1024×768 and274 pairs use
eight smaller aspect/size variants. This is header-level integrity, not evidence
of pixel registration quality.

The dataset cannot enter formal training yet: no defensible train/val/test split
has been frozen, scene/sequence independence and near duplicates are unaudited,
and the locally held split previously described as3141/785/274 is legacy rather
than an established benchmark. The local copy contains no split/meta file.

Raw server inventory: research_protocol/v1/results/m3fd_small_object_audit_v1.json.
Pair audit: research_protocol/v1/results/m3fd_pair_audit_v1.json.
Reproduction: research/audit_m3fd_small_objects.py and audit_m3fd_pairs.py.
