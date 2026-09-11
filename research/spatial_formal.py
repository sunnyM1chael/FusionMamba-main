"""Prepare verified common initialization, or explicitly run the frozen A/B/C queue."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/root/autodl-fs/research_protocol/v1/runs/spatial_formal_corrected_seed42_v1')
SPLIT = Path('/root/autodl-fs/research_protocol/v1/development_splits/msrs')
MODES = ('none', 'independent', 'joint')
EXPECTED = {'train': '9dfa74c7e3ada76bfc64763fefaedc84ec9324cfe3be2d3212c5f23607447b3e',
            'val': 'c234c286a0ea4065011f6bd572541fe400f5bba970a3f91169704448dbe3e940',
            'test': '933b68f7bafa1ca85fd3ad33553a5f6d7275cbc7c2024d79727300a0a210e3f0'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            sorted(list(ROOT.glob('*.py')) + list((ROOT / 'models').glob('*.py')))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    for name, expected in EXPECTED.items():
        assert digest(SPLIT / f'{name}.txt') == expected
    if args.action == 'prepare':
        from models.vmamba_Fusion_efficross import VSSM_Fusion
        BASE.mkdir(parents=True, exist_ok=False)
        canonical, report = {}, {'source_sha256': sources(), 'manifest_sha256': EXPECTED, 'modes': {}}
        for mode in MODES:
            torch.manual_seed(42)
            model = VSSM_Fusion(use_dsdam=True, spatial_mode=mode, weighting_mode='equal')
            state = model.state_dict()
            matched = changed = 0
            for name, tensor in state.items():
                if name in canonical:
                    assert canonical[name].shape == tensor.shape, name
                    matched += 1
                    changed += int(not torch.equal(tensor, canonical[name]))
                    tensor.copy_(canonical[name])
                else:
                    canonical[name] = tensor.clone()
            assert all(torch.equal(t, canonical[n]) for n, t in state.items())
            path = BASE / f'initial_{mode}.pth'
            torch.save({'model': state, 'args': {'spatial_mode': mode, 'weighting_mode': 'equal',
                                               'disable_dsdam': mode == 'none', 'share_encoder_weights': False},
                        'role': 'untrained common initialization, not legacy weights'}, path)
            reloaded = torch.load(path, map_location='cpu', weights_only=False)['model']
            assert all(torch.equal(t, reloaded[n]) for n, t in state.items())
            report['modes'][mode] = {'shared_tensors_verified': matched, 'raw_seed_mismatches_corrected': changed,
                                    'parameters': sum(p.numel() for p in model.parameters()),
                                    'initial_sha256': digest(path)}
            print('INITIALIZATION_VERIFIED', mode, report['modes'][mode], flush=True)
            del model, state, reloaded
        report['policy'] = 'Same-name/shape tensors share exact values and buffers; new tensors use seed42 defaults. A supplies core tensors, B supplies shared enhancer tensors. Optimizers fresh.'
        (BASE / 'preparation.json').write_text(json.dumps(report, indent=2))
        print('PREPARATION_COMPLETE', flush=True)
        return
    assert torch.cuda.is_available()
    report = json.loads((BASE / 'preparation.json').read_text())
    assert sources() == report['source_sha256'], 'Source changed after preparation'
    for mode in MODES:
        assert sources() == report['source_sha256'], 'Source changed during queue'
        initial = BASE / f'initial_{mode}.pth'
        assert digest(initial) == report['modes'][mode]['initial_sha256']
        output = BASE / mode
        output.mkdir(exist_ok=False)
        command = [sys.executable, '-u', str(ROOT / 'train.py'),
                   '--ir_path', '/root/autodl-fs/datasets/MSRS/train/ir', '--vis_path', '/root/autodl-fs/datasets/MSRS/train/vi',
                   '--train_list', str(SPLIT / 'train.txt'), '--val_list', str(SPLIT / 'val.txt'),
                   '--output_dir', str(output), '--pretrained', str(initial), '--spatial_mode', mode,
                   '--weighting_mode', 'equal', '--epochs', '100', '--crop_size', '256', '--batch_size', '4',
                   '--num_workers', '4', '--lr', '0.0001', '--min_lr', '0.000001', '--weight_decay', '0.0001',
                   '--grad_clip', '1.0', '--seed', '42', '--device', '0', '--no-amp']
        (output / 'run_provenance.json').write_text(json.dumps({'command': command, **report}, indent=2))
        with (output / 'train.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (output / 'pid.txt').write_text(str(child.pid))
            print('START', mode, child.pid, flush=True)
            if child.wait():
                raise RuntimeError(f'{mode} failed; queue stopped')
        history = json.loads((output / 'history.json').read_text())
        assert len(history) == 100 and history[-1]['epoch'] == 100
        assert all(h['optimizer_updates'] == 243 and h['skipped_updates'] == 0 for h in history)
        print('COMPLETE', mode, flush=True)
    print('ALL_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
