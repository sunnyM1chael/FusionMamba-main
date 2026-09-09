# Dataset audit findings — protocol v1

The complete read-only audit processed 21,432 paired samples on the server. Its
machine-readable report is `/root/autodl-fs/research_protocol/v1/audit_complete_r2`.

* Counts: MSRS 1,083 train + 361 test; LLVIP 12,025 train + 3,463 test;
  M3FD 4,200 unassigned; M3FD fusion subset 300 unassigned.
* All audited image pairs decoded. Modality dimensions matched.
* No exact or equal-dHash candidates crossed the published MSRS or LLVIP
  train/test boundary. Equal dHash is only a screening heuristic.
* LLVIP contains 42,437 valid person boxes before filtering five zero-width boxes.
  Invalid objects occur in `020118`, `020236`, `090437`, `091180`, `170350`.
  `100030` and `100033` are empty annotations and remain valid negative samples.
* M3FD contains 34,407 valid boxes: People 11,477; Car 18,296; Lamp 2,405;
  Bus 700; Motorcycle 521; Truck 1,008.
* Every one of the 300 `M3FD_Fusion` pairs is pixel-identical to a pair in
  `M3FD_RAW`. Do not count it as independent data or an external test set.
* 523 equal-dHash candidate groups require interpretation as repeated/near-static
  content before making scene-independence claims. There are no such candidates
  crossing published MSRS/LLVIP train/test partitions.

Raw data and XML files were not changed. LLVIP-derived YOLO labels reject the five
invalid objects and record each rejection in `conversion_report.json`.
