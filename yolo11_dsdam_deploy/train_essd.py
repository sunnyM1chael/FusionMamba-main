"""Train ESSD-Head while retaining compatible YOLO11 pretrained features."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import strip_optimizer


ESSD_MAP = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8, 7: 10, 8: 11,
    9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19,
    17: 26, 18: 27, 19: 28, 20: 29, 21: 30, 22: 31,
}
DSDAM_ONLY_MAP = {
    **{i: j for i, j in enumerate((0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13))},
    **{i: j for i, j in zip(range(11, 23), range(14, 26))},
}
P2_ONLY_MAP = {
    **{i: i for i in range(17)},
    **{i: j for i, j in zip(range(17, 23), range(23, 29))},
}
VARIANTS = {
    "essd": ("yolo11-essd.yaml", ESSD_MAP),
    "dsdam_only": ("yolo11-essd-dsdam-only.yaml", DSDAM_ONLY_MAP),
    "p2_only": ("yolo11-essd-p2-only.yaml", P2_ONLY_MAP),
}


class SingleBuildDetectionTrainer(DetectionTrainer):
    """Keep final checkpoint loading isolated from the DSDAM training process."""

    def final_eval(self):
        """Strip checkpoints here; validation runs in a fresh process below."""
        model = self.best if self.best.exists() else None
        ckpt = strip_optimizer(self.last) if self.last.exists() else {}
        if model:
            strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
            LOGGER.info(
                "Final best-checkpoint validation will run in an isolated process "
                "to avoid reinitializing DSDAM after training."
            )


def validate_checkpoint_isolated(trainer):
    """Validate the selected best checkpoint in a clean Python process."""
    checkpoint = trainer.best if trainer.best.exists() else trainer.last
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint available for final validation: {checkpoint}")
    code = (
        "from ultralytics import YOLO; import sys; "
        "YOLO(sys.argv[1]).val(data=sys.argv[2], imgsz=int(sys.argv[3]), "
        "batch=int(sys.argv[4]), device=sys.argv[5], workers=int(sys.argv[6]), "
        "project=sys.argv[7], name=sys.argv[8], exist_ok=True, plots=False)"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(checkpoint),
            str(trainer.args.data),
            str(trainer.args.imgsz),
            str(trainer.args.batch),
            str(trainer.args.device),
            str(trainer.args.workers),
            str(trainer.args.project),
            f"{trainer.args.name}_best_val",
        ],
        check=True,
    )


def transfer_pretrained(model, weights, layer_index_map):
    """Remap standard YOLO11 layers around newly inserted ESSD modules."""
    source = YOLO(str(weights)).model.state_dict()
    target = model.state_dict()
    transferred = {}

    for key, value in source.items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        source_index = int(parts[1])
        if source_index not in layer_index_map:
            continue
        parts[1] = str(layer_index_map[source_index])
        target_key = ".".join(parts)
        if target_key in target and target[target_key].shape == value.shape:
            transferred[target_key] = value

    incompatible = model.load_state_dict(transferred, strict=False)
    print(
        f"Transferred {len(transferred)}/{len(target)} compatible tensors; "
        f"new ESSD tensors={len(incompatible.missing_keys)}"
    )
    return len(transferred)


def train_args(args, model_path):
    """Build one shared Ultralytics override dictionary for all variants."""
    return {
        "model": str(model_path),
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name or f"m3fd_{args.variant}",
        "seed": args.seed,
        # Seeds remain fixed independently. Strict deterministic kernels are
        # opt-in because torchvision DeformConv2d warns that its backward input
        # gradient has no deterministic CUDA implementation.
        "deterministic": args.deterministic,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "cos_lr": True,
        "close_mosaic": 10,
    }


def parse_args():
    root = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Train YOLO11 ESSD-Head on fused M3FD images")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--weights", default=str(root.parent / "yolo11n.pt"))
    parser.add_argument("--variant", choices=("baseline", *VARIANTS), default="essd")
    parser.add_argument("--model", help="Explicit model YAML; uses the selected variant's transfer map")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--project", default="runs/essd")
    parser.add_argument("--name", help="Run name; defaults to m3fd_<variant>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request strict deterministic kernels (not recommended with torchvision DeformConv2d)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.variant == "baseline":
        if args.model:
            raise ValueError("--model is not used with the baseline variant")
        model = YOLO(args.weights)
        model.train(**train_args(args, args.weights))
    else:
        filename, layer_map = VARIANTS[args.variant]
        model_path = args.model or str(Path(__file__).resolve().parent / "ultralytics/cfg/models/11" / filename)
        overrides = train_args(args, model_path)
        # Build the dataset-specific head exactly once. Going through YOLO.train()
        # would rebuild a YAML model after discovering nc and executes a second
        # DSDAM stride-probing forward in the same process. This must also happen
        # before the trainer enables deterministic algorithms: torchvision's CPU
        # deformable-convolution probe is not compatible with that global flag.
        data = check_det_dataset(args.data)
        model = DetectionModel(
            model_path,
            nc=data["nc"],
            ch=data["channels"],
            verbose=True,
        )
        model.task = "detect"
        transfer_pretrained(model, args.weights, layer_map)
        trainer = SingleBuildDetectionTrainer(overrides=overrides)
        trainer.model = model
        trainer.train()
        validate_checkpoint_isolated(trainer)


if __name__ == "__main__":
    main()
