"""Serial two-epoch execution gate, using only frozen development subsets."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = Path('/root/autodl-fs/research_protocol/v1/runs/spatial_smoke_corrected_v1')
base.mkdir(parents=True, exist_ok=False)
split = Path('/root/autodl-fs/research_protocol/v1/development_splits/msrs')
expected = {'train': '9dfa74c7e3ada76bfc64763fefaedc84ec9324cfe3be2d3212c5f23607447b3e',
            'val': 'c234c286a0ea4065011f6bd572541fe400f5bba970a3f91169704448dbe3e940'}
for name, count in [('train', 16), ('val', 4)]:
    source = split / f'{name}.txt'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected[name]
    (base / f'{name}.txt').write_text('\n'.join(source.read_text().splitlines()[:count]) + '\n')
for mode in ('none', 'independent', 'joint'):
    destination = base / mode
    command = [sys.executable, '-u', str(root / 'train.py'), '--ir_path', '/root/autodl-fs/datasets/MSRS/train/ir',
               '--vis_path', '/root/autodl-fs/datasets/MSRS/train/vi', '--train_list', str(base / 'train.txt'),
               '--val_list', str(base / 'val.txt'), '--output_dir', str(destination), '--crop_size', '128',
               '--epochs', '2', '--batch_size', '4', '--num_workers', '2', '--seed', '42', '--no-amp',
               '--weighting_mode', 'equal', '--spatial_mode', mode]
    (base / f'{mode}_command.json').write_text(json.dumps(command))
    print('START', mode, flush=True)
    subprocess.run(command, cwd=root, check=True)
    history = json.loads((destination / 'history.json').read_text())
    assert len(history) == 2 and all(r['optimizer_updates'] == 4 and r['skipped_updates'] == 0 for r in history)
    import torch
    checkpoint = torch.load(destination / 'last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['optimizer'] is not None
    assert all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values())
    print('PASS', mode, flush=True)
    del checkpoint
print('ALL_PASS_EXECUTION_ONLY', flush=True)
