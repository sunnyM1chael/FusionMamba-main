"""Build raw IR/VIS manifests from the existing M3FD YOLO train/val split."""

import argparse
from pathlib import Path


EXTENSIONS = {'.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def by_stem(folder):
    files = {}
    for path in Path(folder).iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            stem = path.stem
            if stem in files:
                raise ValueError(f'Duplicate stem {stem} in {folder}')
            files[stem] = path.name
    return files


def yolo_stems(folder):
    return {path.stem for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ir_path', required=True)
    parser.add_argument('--vis_path', required=True)
    parser.add_argument('--yolo_root', required=True, help='Existing M3FD YOLO root with images/train and images/val')
    parser.add_argument('--output_dir', default='splits/m3fd')
    args = parser.parse_args()

    ir, vis = by_stem(args.ir_path), by_stem(args.vis_path)
    common = set(ir) & set(vis)
    if not common:
        raise ValueError('No raw IR/VIS pairs matched by filename stem')
    train = yolo_stems(Path(args.yolo_root) / 'images' / 'train')
    val = yolo_stems(Path(args.yolo_root) / 'images' / 'val')
    missing = (train | val) - common
    if missing:
        raise ValueError(f'{len(missing)} YOLO images are missing from raw IR/VIS pairs')
    overlap = train & val
    if overlap:
        raise ValueError(f'{len(overlap)} stems appear in both train and val')
    test = common - train - val
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, stems in (('train', train), ('val', val), ('test', test)):
        names = sorted(ir[stem] for stem in stems)
        (output / f'{name}.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        print(f'{name}: {len(names)}')
    print(f'paired_raw: {len(common)}')


if __name__ == '__main__':
    main()
