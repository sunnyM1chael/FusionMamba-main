"""Train ESSD-Head while retaining compatible YOLO11 pretrained features."""

from argparse import ArgumentParser, BooleanOptionalAction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import strip_optimizer, init_seeds


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
    "baseline": ("yolo11.yaml", {i: i for i in range(23)}),
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
        "project=sys.argv[7], name=sys.argv[8], exist_ok=True, plots=False, "
        "split='val', rect=False, half=False)"
    )
    env = os.environ.copy()
    deploy_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = deploy_root + os.pathsep + env.get("PYTHONPATH", "")
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
        cwd=deploy_root,
        env=env,
        check=True,
    )


def transfer_pretrained(model, weights, layer_index_map):
    """Remap standard YOLO11 layers around newly inserted ESSD modules."""
    source = YOLO(str(weights)).model.state_dict()
    target = model.state_dict()
    transferred = {}
    source_digest = hashlib.sha256()

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
            # Hash source names and values, not remapped target indices, so
            # common pretrained tensors can be compared across architectures.
            source_digest.update(key.encode("utf-8"))
            source_digest.update(value.detach().cpu().contiguous().numpy().tobytes())

    incompatible = model.load_state_dict(transferred, strict=False)
    print(
        f"Transferred {len(transferred)}/{len(target)} compatible tensors; "
        f"new or task-specific tensors={len(incompatible.missing_keys)}"
    )
    return {
        "transferred_tensors": len(transferred),
        "target_tensors": len(target),
        "new_tensors": len(incompatible.missing_keys),
        "source_tensor_sha256": source_digest.hexdigest(),
    }


def scaled_model_path(base_path, scale):
    """Return a virtual scaled YAML name resolved by Ultralytics to the base YAML."""
    path = Path(base_path)
    if not path.stem.startswith("yolo11"):
        return path
    suffix = path.stem[len("yolo11"):]
    suffix = re.sub(r"^[nslmx]", "", suffix)
    return path.with_name(f"yolo11{scale}{suffix}{path.suffix}")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "amp": args.amp,
        "cache": False,
        "rect": False,
        "plots": False,
        "pretrained": False,
        "cos_lr": True,
        "close_mosaic": 10,
    }


def parse_args():
    root = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Train YOLO11 ESSD-Head on fused M3FD images")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--scale", choices=("n", "s"), default="s")
    parser.add_argument("--weights", help="Pretrained checkpoint; defaults to yolo11<scale>.pt")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="essd")
    parser.add_argument("--model", help="Explicit model YAML; uses the selected variant's transfer map")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping for fixed-epoch ablation")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--amp", action=BooleanOptionalAction, default=True)
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
    root = Path(__file__).resolve().parent
    weights = Path(args.weights) if args.weights else root.parent / f"yolo11{args.scale}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {weights}")
    # New modules are initialized before the trainer exists.
    init_seeds(args.seed, deterministic=False)
    filename, layer_map = VARIANTS[args.variant]
    base_model_path = Path(args.model) if args.model else root / "ultralytics/cfg/models/11" / filename
    model_path = scaled_model_path(base_model_path, args.scale)
    overrides = train_args(args, model_path)
    # Build the dataset-specific head exactly once. Going through YOLO.train()
    # would rebuild a YAML model after discovering nc and execute a second
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
    initialization = transfer_pretrained(model, weights, layer_map)
    trainer = SingleBuildDetectionTrainer(overrides=overrides)
    trainer.model = model
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "variant": args.variant,
        "scale": args.scale,
        "weights": str(weights.resolve()),
        "weights_sha256": file_sha256(weights),
        "seed": args.seed,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "full_initialized_model_sha256": model_state_sha256(model),
        "train_args": train_args(args, model_path),
        **initialization,
    }
    (trainer.save_dir / "initialization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    trainer.train()
    validate_checkpoint_isolated(trainer)


if __name__ == "__main__":
    main()
