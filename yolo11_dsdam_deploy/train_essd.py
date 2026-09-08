"""Train ESSD-Head while retaining compatible YOLO11 pretrained features."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


LAYER_INDEX_MAP = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8, 7: 10, 8: 11,
    9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19,
    17: 26, 18: 27, 19: 28, 20: 29, 21: 30, 22: 31,
}


def transfer_pretrained(model, weights):
    """Remap standard YOLO11 layers around newly inserted ESSD modules."""
    source = YOLO(str(weights)).model.state_dict()
    target = model.model.state_dict()
    transferred = {}

    for key, value in source.items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        source_index = int(parts[1])
        if source_index not in LAYER_INDEX_MAP:
            continue
        parts[1] = str(LAYER_INDEX_MAP[source_index])
        target_key = ".".join(parts)
        if target_key in target and target[target_key].shape == value.shape:
            transferred[target_key] = value

    incompatible = model.model.load_state_dict(transferred, strict=False)
    print(
        f"Transferred {len(transferred)}/{len(target)} compatible tensors; "
        f"new ESSD tensors={len(incompatible.missing_keys)}"
    )
    return len(transferred)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Train YOLO11 ESSD-Head on fused M3FD images")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--weights", default=str(root.parent / "yolo11n.pt"))
    parser.add_argument(
        "--model",
        default=str(root / "ultralytics/cfg/models/11/yolo11-essd.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--project", default="runs/essd")
    parser.add_argument("--name", default="m3fd")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    transfer_pretrained(model, args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=True,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
