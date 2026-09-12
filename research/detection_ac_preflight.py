"""Audit frozen LLVIP development split and time A/C full-resolution export.

No detector training or test-set inference. Outputs are versioned and exclusive.
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fused_dataset import load_model, infer_full
from split_manifest import read_manifest

ROOT = Path('/root/autodl-fs/research_protocol/v1')
OUT = ROOT / 'results/detection_ac_preflight_v1'
DATA = Path('/root/autodl-fs/datasets/LLVIP')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = ROOT / 'development_splits/llvip'
    expected = {'train': '7154252d9ce730dc9e50d925433084e1489bb535cca628b5f0f4c0d83b8ef5db',
                'val': 'd0da3502753369bdd3e3777ae2b23a68e61b71304c2e9b04881046ed109b30f0'}
    names = {}
    for split, digest in expected.items():
        assert hashlib.sha256((manifest / f'{split}.txt').read_bytes()).hexdigest() == digest
        names[split] = read_manifest(manifest / f'{split}.txt')
    assert not set(names['train']) & set(names['val'])
    membership = {name: split for split, values in names.items() for name in values}
    groups = defaultdict(list)
    seen = set()
    for line in (ROOT / 'audit_complete_r2/samples.jsonl').open():
        row = json.loads(line)
        if row['dataset'] != 'LLVIP' or row['partition'] != 'train':
            continue
        name = row['name']
        assert name in membership
        seen.add(name)
        for modality in ('ir', 'vis'):
            for key in ('pixel_sha256', 'dhash'):
                groups[(modality, key, row[modality][key])].append(name)
        assert int(row['xml_size'][0]) == row['ir']['width']
        assert int(row['xml_size'][1]) == row['ir']['height']
        split = membership[name]
        assert (ROOT / f'llvip_yolo/infrared/labels/{split}/{Path(name).stem}.txt').is_file()
    assert seen == set(membership)
    crossings = [{'modality': k[0], 'kind': k[1], 'names': values}
                 for k, values in groups.items()
                 if len({membership[n] for n in values}) > 1]
    report = {'counts': {s: len(v) for s, v in names.items()}, 'manifest_sha256': expected,
              'cross_split_candidates': crossings,
              'limitations': 'Historical decoded-hash audit reused; dHash equality is not a comprehensive near-duplicate check. Prefix groups are not verified scenes.',
              'export_policy': 'FP32, full native resolution, grayscale replicated by detector loader, PNG, clamp [0,1], no per-image normalization',
              'models': {}}
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2))
    if crossings:
        raise RuntimeError('Cross-split duplicate candidates require review before training')
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    # Spread fixed training samples over the ordered list; never select on output.
    samples = [names['train'][i] for i in (0, len(names['train']) // 3,
                                          2 * len(names['train']) // 3, len(names['train']) - 1)]
    for mode in ('none', 'joint'):
        checkpoint = ROOT / f'runs/spatial_formal_corrected_seed42_v1/{mode}/best.pth'
        model = load_model(checkpoint, device, False, False)
        folder = OUT / mode
        folder.mkdir()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for name in samples:
            ir = cv2.imread(str(DATA / 'infrared/train' / name), cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(str(DATA / 'visible/train' / name), cv2.IMREAD_COLOR)
            assert ir is not None and color is not None
            vis = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            assert ir.shape == vis.shape
            torch.cuda.synchronize()
            start = time.monotonic()
            output = infer_full(model, ir.astype(np.float32) / 255,
                                vis.astype(np.float32) / 255, device, False)
            torch.cuda.synchronize()
            assert np.isfinite(output).all()
            dest = folder / (Path(name).stem + '.png')
            assert cv2.imwrite(str(dest), (output * 255).round().clip(0, 255).astype(np.uint8))
            timings.append({'name': name, 'seconds_with_png': time.monotonic() - start,
                            'bytes': dest.stat().st_size, 'shape': list(ir.shape),
                            'clipped_fraction': float(np.mean((output < 0) | (output > 1)))})
        report['models'][mode] = {'samples': timings,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (OUT / 'audit.json').write_text(json.dumps(report, indent=2))
        print(mode, report['models'][mode], flush=True)
        del model
        torch.cuda.empty_cache()
    print('PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
