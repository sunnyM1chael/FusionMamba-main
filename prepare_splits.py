"""Create reproducible paired-image splits, optionally grouped by scene/sequence."""

import argparse
import random
import re
from pathlib import Path


EXTENSIONS = {'.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def image_names(folder):
    return {p.name for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}


def split_groups(names, train_ratio, val_ratio, seed, group_regex):
    pattern = re.compile(group_regex) if group_regex else None
    groups = {}
    for name in sorted(names):
        match = pattern.search(name) if pattern else None
        key = match.group(1) if match else name
        groups.setdefault(key, []).append(name)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    total = len(names)
    targets = [round(total * train_ratio), round(total * val_ratio)]
    splits = [[], [], []]
    for key in keys:
        if len(splits[0]) < targets[0]:
            index = 0
        elif len(splits[1]) < targets[1]:
            index = 1
        else:
            index = 2
        splits[index].extend(groups[key])
    return [sorted(part) for part in splits]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ir_path', required=True)
    parser.add_argument('--vis_path', required=True)
    parser.add_argument('--output_dir', default='splits')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--group_regex',
        help='Regex whose first capture group is a scene/sequence ID; grouped files stay together',
    )
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('Ratios must satisfy train>0, val>=0 and train+val<1')

    ir_names = image_names(args.ir_path)
    vis_names = image_names(args.vis_path)
    names = ir_names & vis_names
    if not names:
        raise ValueError('No filename-matched IR/VIS pairs found')
    if ir_names != vis_names:
        print(f'Warning: unmatched files: IR-only={len(ir_names-vis_names)}, VIS-only={len(vis_names-ir_names)}')
    train, val, test = split_groups(
        names, args.train_ratio, args.val_ratio, args.seed, args.group_regex
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name, values in [('train', train), ('val', val), ('test', test)]:
        (output / f'{split_name}.txt').write_text('\n'.join(values) + '\n', encoding='utf-8')
        print(f'{split_name}: {len(values)}')
    if not args.group_regex:
        print('Warning: split is image-level. Use --group_regex when filenames contain scene/sequence IDs.')


if __name__ == '__main__':
    main()
