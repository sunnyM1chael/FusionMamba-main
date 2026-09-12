"""Serial A/C export and detector smoke gates; never starts formal training."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from generate_fused_dataset import infer_full, load_model
from split_manifest import read_manifest

ROOT = Path('/root/autodl-fs/research_protocol/v1')
OUT = ROOT / 'runs/detection_ac_corrected_seed42_v1'
DATA = Path('/root/autodl-fs/datasets/LLVIP')


def save_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


def export(mode, stage, audit):
    folder = OUT / stage / mode
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / f'runs/spatial_formal_corrected_seed42_v1/{mode}/best.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == audit['models'][mode]['checkpoint_sha256']
    device = torch.device('cuda:0')
    model = load_model(checkpoint, device, False, False)
    start = time.monotonic()
    rows = []
    for split in ('train', 'val'):
        manifest = ROOT / f'development_splits/llvip/{split}.txt'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == audit['manifest_sha256'][split]
        names = read_manifest(manifest)
        if stage == 'smoke':
            names = read_manifest(ROOT / f'development_splits/llvip/smoke_{split}.txt')
        images, labels = folder / 'images' / split, folder / 'labels' / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names, 1):
            ir = cv2.imread(str(DATA / 'infrared/train' / name), cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(str(DATA / 'visible/train' / name), cv2.IMREAD_COLOR)
            if ir is None or color is None:
                raise ValueError(name)
            vis = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            assert ir.shape == vis.shape
            target = images / (Path(name).stem + '.png')
            label = ROOT / f'llvip_yolo/infrared/labels/{split}/{Path(name).stem}.txt'
            label_target = labels / label.name
            existing = cv2.imread(str(target), cv2.IMREAD_UNCHANGED) if target.is_file() else None
            if existing is not None:
                if existing.shape != ir.shape or existing.dtype != np.uint8:
                    raise ValueError(f'Invalid existing output requiring review: {target}')
                clipped_fraction = None
            else:
                output = infer_full(model, ir.astype(np.float32) / 255,
                                    vis.astype(np.float32) / 255, device, False)
                if not np.isfinite(output).all():
                    raise FloatingPointError(name)
                clipped_fraction = float(np.mean((output < 0) | (output > 1)))
                temporary = target.with_suffix('.partial.png')
                assert cv2.imwrite(str(temporary), (output * 255).round().clip(0, 255).astype(np.uint8))
                os.replace(temporary, target)
            if label_target.is_file():
                assert hashlib.sha256(label_target.read_bytes()).digest() == hashlib.sha256(label.read_bytes()).digest()
            else:
                shutil.copyfile(label, label_target)
            rows.append({'split': split, 'source': name, 'output': target.name,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'label_sha256': hashlib.sha256(label.read_bytes()).hexdigest(),
                         'clipped_fraction': clipped_fraction,
                         'reused_verified_existing': existing is not None})
            if index % 250 == 0 or index == len(names):
                status = {'mode': mode, 'split': split, 'done': index, 'total': len(names),
                          'elapsed_seconds': time.monotonic() - start}
                save_json(folder / 'progress.json', status)
                print(stage, status, flush=True)
        assert len(list(images.glob('*.png'))) == len(names)
    # No test alias: official test must not be evaluated during development.
    (folder / 'data.yaml').write_text(f'path: {folder}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n')
    save_json(folder / 'export.json', {'policy': audit['export_policy'],
        'checkpoint_sha256': audit['models'][mode]['checkpoint_sha256'],
        'rows': rows, 'elapsed_seconds': time.monotonic() - start})
    del model
    torch.cuda.empty_cache()


def queue():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / 'results/detection_ac_preflight_v1/audit.json').read_text())
    assert not audit['cross_split_candidates'] and set(audit['models']) == {'none', 'joint'}
    assert shutil.disk_usage(OUT).free > 30 * 2**30
    provenance = {'preflight': audit, 'source_hashes': {
        str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), SOURCE / 'research/train_detection_ac.py',
                  SOURCE / 'generate_fused_dataset.py', SOURCE / 'DSDAM.py',
                  SOURCE / 'models/vmamba_Fusion_efficross.py']}}
    save_json(OUT / ('resume_provenance.json' if (OUT / 'provenance.json').exists() else 'provenance.json'), provenance)
    for stage in ('smoke', 'full'):
        for mode in ('none', 'joint'):
            if (OUT / stage / mode / 'export.json').is_file():
                continue
            save_json(OUT / 'status.json', {'phase': 'export', 'stage': stage, 'mode': mode})
            subprocess.run([sys.executable, str(Path(__file__)), '--export', mode, '--stage', stage], check=True)
        if stage == 'smoke':
            for mode in ('none', 'joint'):
                if (OUT / stage / mode / 'training_complete.json').is_file():
                    continue
                save_json(OUT / 'status.json', {'phase': 'smoke_train', 'mode': mode})
                subprocess.run([sys.executable, str(SOURCE / 'research/train_detection_ac.py'),
                                '--mode', mode, '--stage', 'smoke'], check=True)
    save_json(OUT / 'status.json', {'phase': 'READY_FOR_FORMAL_REVIEW',
        'note': 'Both smoke gates and full exports complete. No formal detector training launched.'})
    print('READY_FOR_FORMAL_REVIEW', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', choices=('none', 'joint'))
    parser.add_argument('--stage', choices=('smoke', 'full'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    try:
        if args.export:
            export(args.export, args.stage, json.loads((ROOT / 'results/detection_ac_preflight_v1/audit.json').read_text()))
        else:
            queue()
    except Exception as exc:
        if OUT.exists() and not args.export:
            save_json(OUT / 'failure.json', {'error': repr(exc), 'time': time.time()})
        raise
