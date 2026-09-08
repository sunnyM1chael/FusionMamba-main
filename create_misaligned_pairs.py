"""Create controlled VIS perturbations for SACAFM robustness experiments."""

import argparse
import json
from pathlib import Path

import cv2


EXTENSIONS = {'.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--translate_x', type=float, default=0.0)
    parser.add_argument('--translate_y', type=float, default=0.0)
    parser.add_argument('--rotate', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--border', choices=('reflect', 'replicate', 'constant'), default='reflect')
    args = parser.parse_args()

    border_modes = {
        'reflect': cv2.BORDER_REFLECT_101,
        'replicate': cv2.BORDER_REPLICATE,
        'constant': cv2.BORDER_CONSTANT,
    }
    source = Path(args.vis_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Failed to load {path}')
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), args.rotate, args.scale)
        matrix[0, 2] += args.translate_x
        matrix[1, 2] += args.translate_y
        transformed = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=border_modes[args.border],
        )
        cv2.imwrite(str(output / path.name), transformed)
        if index % 100 == 0 or index == len(files):
            print(f'{index}/{len(files)}')
    (output / 'perturbation.json').write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
