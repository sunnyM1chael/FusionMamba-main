"""Create sequence-disjoint KAIST train/val/test manifests without copying images."""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaist_root', required=True)
    parser.add_argument('--output_dir', default='splits/kaist')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('train_ratio must be positive and train_ratio+val_ratio must be below 1')

    root = Path(args.kaist_root)
    pairs = []
    groups = {}
    for lwir in root.glob('**/lwir/*'):
        if not lwir.is_file() or lwir.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        visible = lwir.parent.parent / 'visible' / lwir.name
        if not visible.is_file():
            continue
        key = lwir.relative_to(root).as_posix()
        group = lwir.parent.parent.relative_to(root).as_posix()
        pairs.append(key)
        groups.setdefault(group, []).append(key)
    if not pairs:
        raise ValueError(f'No KAIST pairs found under {root}')

    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    targets = [round(len(pairs) * args.train_ratio), round(len(pairs) * args.val_ratio)]
    splits = [[], [], []]
    for key in keys:
        index = 0 if len(splits[0]) < targets[0] else 1 if len(splits[1]) < targets[1] else 2
        splits[index].extend(groups[key])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in zip(('train', 'val', 'test'), splits):
        (output / f'{name}.txt').write_text('\n'.join(sorted(values)) + '\n', encoding='utf-8')
        print(f'{name}: {len(values)}')


if __name__ == '__main__':
    main()
