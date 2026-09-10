"""Launch the frozen first MSRS development run and record its provenance."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--weighting_mode', choices=('equal', 'learned', 'acgaw'), default='acgaw')
    parser.add_argument('--wait', action='store_true', help='Wait for completion and propagate failure for a serial queue')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = Path('/root/autodl-fs/research_protocol/v1/development_splits/msrs')
    expected = {
        'train': '9dfa74c7e3ada76bfc64763fefaedc84ec9324cfe3be2d3212c5f23607447b3e',
        'val': 'c234c286a0ea4065011f6bd572541fe400f5bba970a3f91169704448dbe3e940',
        'test': '933b68f7bafa1ca85fd3ad33553a5f6d7275cbc7c2024d79727300a0a210e3f0',
    }
    for split, digest in expected.items():
        if hashlib.sha256((protocol / f'{split}.txt').read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen {split} manifest changed')
    if args.resume:
        if not args.output.is_dir() or args.resume.parent != args.output:
            raise ValueError('Resume checkpoint must be inside the existing output directory')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(root / 'train.py'),
               '--ir_path', '/root/autodl-fs/datasets/MSRS/train/ir',
               '--vis_path', '/root/autodl-fs/datasets/MSRS/train/vi',
               '--train_list', str(protocol / 'train.txt'),
               '--val_list', str(protocol / 'val.txt'),
               '--output_dir', str(args.output), '--crop_size', '256',
               '--batch_size', '4', '--num_workers', '4', '--epochs', '100',
               '--lr', '0.0001', '--min_lr', '0.000001', '--weight_decay', '0.0001',
               '--grad_clip', '1.0', '--seed', '42', '--device', '0', '--no-amp',
               '--weighting_mode', args.weighting_mode, '--log_interval', '10']
    if args.resume:
        command.extend(['--resume', str(args.resume)])
    sources = {}
    for folder in (root, root / 'models'):
        for path in folder.glob('*.py'):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'commit': args.commit, 'source_sha256': sources, 'manifest_sha256': expected,
              'command': command, 'role': 'first full MSRS development run; no test evaluation',
              'selection': 'minimum full-resolution validation fusion loss',
              'python': sys.version}
    if args.resume:
        with (args.output / 'resume_events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
    else:
        (args.output / 'run_provenance.json').write_text(json.dumps(record, indent=2))
        with (args.output / 'environment.txt').open('w') as handle:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
    with (args.output / 'train.log').open('a' if args.resume else 'w') as handle:
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (args.output / 'pid.txt').write_text(str(child.pid))
    print(json.dumps({'pid': child.pid, 'output': str(args.output)}))
    if args.wait:
        code = child.wait()
        if code:
            raise SystemExit(code)
        history = json.loads((args.output / 'history.json').read_text())
        if len(history) != 100 or history[-1]['epoch'] != 100:
            raise RuntimeError('Training exited without completing the frozen 100 epochs')


if __name__ == '__main__':
    main()
