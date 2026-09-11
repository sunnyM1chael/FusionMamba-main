"""Read-only summary of the three server weighting runs; run on server via stdin."""
import json
import math
import statistics
import sys
from pathlib import Path

root = Path('/root/autodl-fs/research_protocol/v1/runs')
names = {
    'acgaw': 'msrs_sacafm_acgaw_seed42_e2cbf9f',
    'equal': 'msrs_sacafm_equal_seed42_controls_v1',
    'learned': 'msrs_sacafm_learned_seed42_controls_v1',
}
records = {}
for mode, name in names.items():
    folder = root / name
    history = json.loads((folder / 'history.json').read_text())
    provenance = json.loads((folder / 'run_provenance.json').read_text())
    args = json.loads((folder / 'train_args.json').read_text())
    best = min(history, key=lambda x: x['val_total'])
    tail = history[-10:]
    records[mode] = {'epochs': len(history), 'best_epoch': best['epoch'],
        'best_val': best['val_total'], 'last_val': history[-1]['val_total'],
        'tail10_mean': statistics.mean(x['val_total'] for x in tail),
        'tail10_std': statistics.pstdev(x['val_total'] for x in tail),
        'tail10_components': {k: statistics.mean(x[k] for x in tail)
                              for k in ('val_intensity', 'val_ssim', 'val_gradient')},
        'updates': sum(x['optimizer_updates'] for x in history),
        'skips': sum(x['skipped_updates'] for x in history),
        'finite_val': all(math.isfinite(x['val_total']) for x in history),
        'checkpoint_bytes': {n: (folder / n).stat().st_size for n in ('best.pth', 'last.pth')},
        'args': args, 'source_sha256': provenance.get('source_sha256'),
        'manifest_sha256': provenance.get('manifest_sha256')}
if '--verify' in sys.argv:
    import torch
    torch.set_num_threads(2)
    for mode, name in names.items():
        if records[mode]['epochs'] != 100:
            raise RuntimeError(f'{mode} not finished; do not inspect an active checkpoint')
        checks = {}
        for filename in ('best.pth', 'last.pth'):
            checkpoint = torch.load(root / name / filename, map_location='cpu', weights_only=False)
            state = checkpoint['model']
            checks[filename] = {
                'epoch_zero_based': checkpoint['epoch'],
                'tensor_count': len(state),
                'finite_model': all(bool(torch.isfinite(v).all()) for v in state.values()
                                    if torch.is_tensor(v)),
                'resumable': checkpoint.get('optimizer') is not None,
            }
            del state, checkpoint
        records[mode]['checkpoint_checks'] = checks
if '--compact' in sys.argv:
    for value in records.values():
        for field in ('args', 'source_sha256', 'manifest_sha256'):
            value.pop(field)
print(json.dumps(records, indent=2))
