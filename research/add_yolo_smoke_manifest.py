"""Add pilot-only file lists to an already generated YOLO dataset view."""
import argparse
from pathlib import Path

from split_manifest import read_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root', type=Path, required=True)
    p.add_argument('--protocol_dir', type=Path, required=True)
    args = p.parse_args()
    for split in ('train','val'):
        names = read_manifest(args.protocol_dir/f'smoke_{split}.txt')
        # Keep the YOLO view path so its images/ -> labels/ mapping is preserved.
        paths = [(args.dataset_root/'images'/split/name).absolute().as_posix() for name in names]
        if not all(Path(path).is_file() for path in paths):
            raise FileNotFoundError(f'Missing smoke {split} images')
        (args.dataset_root/f'smoke_{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    yaml = (f'path: {args.dataset_root.resolve().as_posix()}\n'
            'train: smoke_train.txt\nval: smoke_val.txt\ntest: smoke_val.txt\n'
            'names:\n  0: person\n')
    (args.dataset_root/'LLVIP-smoke.yaml').write_text(yaml,encoding='utf-8')
    print(args.dataset_root/'LLVIP-smoke.yaml')


if __name__ == '__main__':
    main()
