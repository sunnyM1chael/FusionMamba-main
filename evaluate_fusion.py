"""Evaluate fused images with deterministic source-referenced fusion metrics."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from split_manifest import read_manifest, paired_paths, manifest_sha256


EXTENSIONS = {'.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def scan(folder):
    return {p.relative_to(folder).as_posix(): p for p in Path(folder).rglob('*')
            if p.is_file() and p.suffix.lower() in EXTENSIONS}


def evaluation_samples(ir_path, vis_path, fused_path, split_file=None):
    fused = scan(fused_path)
    if split_file:
        names = read_manifest(split_file)
        pairs = paired_paths(ir_path, vis_path, names)
        if set(fused) != set(names):
            raise ValueError(f'Fused output does not match manifest: '
                             f'missing={sorted(set(names)-set(fused))[:5]}, '
                             f'extra={sorted(set(fused)-set(names))[:5]}')
        return [(name, pair[0], pair[1], fused[name]) for name, pair in zip(names, pairs)]
    ir, vis = scan(ir_path), scan(vis_path)
    if not ir or set(ir) != set(vis) or set(ir) != set(fused):
        raise ValueError('IR/VIS/fused sample sets must match exactly; use --split_file for subsets')
    return [(name, ir[name], vis[name], fused[name]) for name in sorted(ir)]


def entropy(image):
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probability = hist[hist > 0] / hist.sum()
    return float(-(probability * np.log2(probability)).sum())


def spatial_frequency(image):
    x = image.astype(np.float64)
    row = np.sqrt(np.mean(np.diff(x, axis=0) ** 2)) if x.shape[0] > 1 else 0.0
    col = np.sqrt(np.mean(np.diff(x, axis=1) ** 2)) if x.shape[1] > 1 else 0.0
    return float(np.sqrt(row ** 2 + col ** 2))


def average_gradient(image):
    x = image.astype(np.float64)
    gx = np.diff(x, axis=1)[:-1, :]
    gy = np.diff(x, axis=0)[:, :-1]
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))


def mutual_information(first, second):
    joint, _, _ = np.histogram2d(first.ravel(), second.ravel(), bins=256, range=((0, 256), (0, 256)))
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    independent = px @ py
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / independent[valid])))


def source_ssim(first, second):
    try:
        from skimage.metrics import structural_similarity
    except (ImportError, ValueError):
        # Some environments have an ABI-incompatible scikit-image build.
        return None
    return float(structural_similarity(first, second, data_range=255))


def metrics(ir, vis, fused):
    ssim_ir = source_ssim(ir, fused)
    ssim_vis = source_ssim(vis, fused)
    return {
        'EN': entropy(fused),
        'SD': float(np.std(fused.astype(np.float64))),
        'SF': spatial_frequency(fused),
        'AG': average_gradient(fused),
        'MI': mutual_information(ir, fused) + mutual_information(vis, fused),
        'SSIM': (ssim_ir + ssim_vis) / 2 if ssim_ir is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ir_path', required=True)
    parser.add_argument('--vis_path', required=True)
    parser.add_argument('--fused_path', required=True)
    parser.add_argument('--split_file', help='Frozen manifest; missing or extra fused images are errors')
    parser.add_argument('--output', default='fusion_metrics.json')
    args = parser.parse_args()

    samples = evaluation_samples(args.ir_path, args.vis_path, args.fused_path, args.split_file)
    rows = []
    for name, ir, vis, fused in samples:
        images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in (ir, vis, fused)]
        if any(image is None for image in images):
            raise ValueError(f'Failed to load {name}')
        if len({image.shape for image in images}) != 1:
            raise ValueError(f'Shape mismatch for {name}')
        rows.append({'name': name, **metrics(*images)})

    keys = ('EN', 'SD', 'SF', 'AG', 'MI', 'SSIM')
    summary = {
        key: {
            'mean': float(np.mean([row[key] for row in rows if row[key] is not None])),
            'std': float(np.std([row[key] for row in rows if row[key] is not None])),
        }
        for key in keys if any(row[key] is not None for row in rows)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'count': len(rows), 'summary': summary,
                                 'manifest_sha256': manifest_sha256(args.split_file) if args.split_file else None,
                                 'protocol': 'strict-v1; OpenCV grayscale; 8-bit; source-mean SSIM'}, indent=2), encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('name', *keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'count': len(rows), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
