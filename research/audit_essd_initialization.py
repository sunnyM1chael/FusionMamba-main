"""Build every detector variant and audit its pretrained initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-root", type=Path, default=Path("yolo11_dsdam_deploy"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scale", choices=("n", "s"), required=True)
    parser.add_argument("--nc", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    deploy_root = args.deploy_root.resolve()
    sys.path.insert(0, str(deploy_root))
    from train_essd import VARIANTS, model_state_sha256, scaled_model_path, transfer_pretrained
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds

    cfg_root = deploy_root / "ultralytics/cfg/models/11"
    rows = []
    for variant, (filename, layer_map) in VARIANTS.items():
        init_seeds(args.seed, deterministic=False)
        model_path = scaled_model_path(cfg_root / filename, args.scale)
        model = DetectionModel(model_path, nc=args.nc, ch=3, verbose=False)
        audit = transfer_pretrained(model, args.weights.resolve(), layer_map)
        rows.append(
            {
                "variant": variant,
                "scale": model.yaml.get("scale"),
                "parameters": sum(p.numel() for p in model.parameters()),
                "full_initialized_model_sha256": model_state_sha256(model),
                **audit,
            }
        )

    common_hashes = {row["source_tensor_sha256"] for row in rows}
    common_counts = {row["transferred_tensors"] for row in rows}
    checkpoint_digest = hashlib.sha256()
    with args.weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    result = {
        "weights": str(args.weights.resolve()),
        "weights_sha256": checkpoint_digest.hexdigest(),
        "nc": args.nc,
        "seed": args.seed,
        "variants": rows,
        "common_pretrained_tensor_count": len(common_counts) == 1,
        "common_pretrained_source_hash": len(common_hashes) == 1,
        "status": "pass" if len(common_counts) == len(common_hashes) == 1 else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
