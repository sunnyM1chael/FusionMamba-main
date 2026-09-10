"""Recover a validated weights-only best checkpoint from a resumable checkpoint."""
import argparse
import os
from pathlib import Path
import zipfile

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    payload = {
        'epoch': checkpoint['epoch'],
        'model': checkpoint['model'],
        'optimizer': None,
        'scheduler': None,
        'scaler': None,
        'best_val': checkpoint['best_val'],
        'history': checkpoint.get('history', []),
        'args': checkpoint.get('args', {}),
        'checkpoint_type': 'weights',
        'rng': None,
        'recovered_from': str(args.source.resolve()),
    }
    temporary = args.output.with_name(f'.{args.output.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        with zipfile.ZipFile(temporary, 'r') as archive:
            if not archive.namelist():
                raise OSError(f'Empty checkpoint archive: {temporary}')
        with temporary.open('r+b') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    verified = torch.load(args.output, map_location='cpu', weights_only=False)
    print({'epoch': verified['epoch'] + 1, 'best_val': verified['best_val'],
           'checkpoint_type': verified['checkpoint_type'], 'bytes': args.output.stat().st_size})


if __name__ == '__main__':
    main()
