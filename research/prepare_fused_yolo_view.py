"""Create a strict YOLO view for generated fused images without copying labels."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_manifest import read_manifest


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    os.symlink(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_root", type=Path, required=True)
    parser.add_argument("--label_root", type=Path, required=True)
    parser.add_argument("--protocol_dir", type=Path, required=True)
    parser.add_argument("--output_yaml", type=Path)
    args = parser.parse_args()

    file_lists = {}
    for split in ("train", "val"):
        names = read_manifest(args.protocol_dir / f"smoke_{split}.txt")
        expected = set(names)
        image_dir = args.fused_root / "images" / split
        actual = {p.name for p in image_dir.iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError(
                f"Fused {split} set mismatch: missing={sorted(expected - actual)[:5]}, "
                f"extra={sorted(actual - expected)[:5]}"
            )
        paths = []
        for name in names:
            image = (image_dir / name).absolute()
            label = args.label_root / split / f"{Path(name).stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            link_file(label, args.fused_root / "labels" / split / label.name)
            paths.append(image.as_posix())
        file_list = args.fused_root / f"{split}.txt"
        file_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
        file_lists[split] = file_list.name

    output_yaml = args.output_yaml or args.fused_root / "LLVIP-fused-smoke.yaml"
    output_yaml.write_text(
        f"path: {args.fused_root.absolute().as_posix()}\n"
        f"train: {file_lists['train']}\n"
        f"val: {file_lists['val']}\n"
        f"test: {file_lists['val']}\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    print(output_yaml)


if __name__ == "__main__":
    main()
